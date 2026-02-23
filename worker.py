import os
import time
import requests
import numpy as np
import cv2

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

RAW_BUCKET = "hdr_raw"
OUT_BUCKET = "hdr_output"
POLL_SECONDS = 2

# Keep this conservative for Railway stability.
# If you upgrade Railway memory later, set this to 2200–2600.
MAX_DIM = int(os.environ.get("MAX_DIM", "1600"))


def headers_json():
    return {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def headers_storage():
    return {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
    }


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rest_get(table: str, params: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.get(url, headers=headers_json(), params=params, timeout=60)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"REST GET {table} failed {r.status_code}: {r.text}")
    return r.json()


def rest_patch(table: str, where_params: dict, payload: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.patch(url, headers=headers_json(), params=where_params, json=payload, timeout=60)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"REST PATCH {table} failed {r.status_code}: {r.text}")


def pick_job():
    rows = rest_get(
        "hdr_jobs",
        {
            "select": "id,set_id,status,attempts,locked_at",
            "status": "eq.queued",
            "locked_at": "is.null",
            "limit": "1",
        },
    )
    return rows[0] if rows else None


def get_set(set_id: str):
    rows = rest_get(
        "hdr_sets",
        {
            "select": "id,order_id,file_under_id,file_base_id,file_over_id,status",
            "id": f"eq.{set_id}",
            "limit": "1",
        },
    )
    if not rows:
        raise RuntimeError("Set not found")
    return rows[0]


def get_file(file_id: str):
    rows = rest_get(
        "hdr_files",
        {
            "select": "id,storage_path",
            "id": f"eq.{file_id}",
            "limit": "1",
        },
    )
    if not rows:
        raise RuntimeError("File not found")
    return rows[0]


def normalize_storage_path(bucket: str, path: str):
    path = (path or "").lstrip("/")
    if path.startswith(f"{bucket}/"):
        path = path[len(bucket) + 1 :]
    return path


def downscale_max(img, max_dim: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def storage_download(bucket: str, rel_path: str):
    rel_path = normalize_storage_path(bucket, rel_path)
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{rel_path}"
    print("Downloading:", f"{bucket}/{rel_path}")

    r = requests.get(url, headers=headers_storage(), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed {r.status_code}: {r.text}")

    buf = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Image decode failed")

    return downscale_max(img, MAX_DIM)


def storage_upload(bucket: str, rel_path: str, img_bgr):
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Encode failed")

    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{rel_path}"
    print("Uploading:", f"{bucket}/{rel_path}")

    r = requests.post(
        url,
        headers={
            **headers_storage(),
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        },
        data=enc.tobytes(),
        timeout=180,
    )
    if r.status_code // 100 != 2:
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text}")


# -------------------------
# CLEAN, BRIGHT HDR MERGE
# -------------------------
def merge_mertens_clean(under, base, over):
    # Same size
    h, w = base.shape[:2]
    under = cv2.resize(under, (w, h))
    over = cv2.resize(over, (w, h))

    imgs = [under, base, over]
    imgs_f = [i.astype(np.float32) / 255.0 for i in imgs]

    mertens = cv2.createMergeMertens()
    fused = mertens.process(imgs_f)
    fused = np.clip(fused, 0.0, 1.0)

    # Prevent “black” output: normalize to a real display range
    mx = float(fused.max())
    if mx > 1e-6:
        fused = fused / mx

    # Bright/airy look: lift midtones slightly
    fused = np.power(fused, 1.0 / 2.0)  # lighter than 2.2 gamma

    out = (fused * 255.0).clip(0, 255).astype(np.uint8)
    return out


def tonal_clean_bright(img_bgr):
    """
    Luxury clean toning:
    - raise exposure using percentile scaling
    - lift shadows lightly
    - protect highlights
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    Lf = L.astype(np.float32)

    # Scale so 98th percentile maps near 245 (bright but not blown)
    p98 = np.percentile(Lf, 98)
    if p98 > 1:
        scale = 245.0 / p98
        Lf = np.clip(Lf * scale, 0, 255)

    # Gentle shadow lift: curve
    # y = x^(0.92) makes darks brighter without nuking highlights
    Lf = np.power(Lf / 255.0, 0.92) * 255.0

    L = np.clip(Lf, 0, 255).astype(np.uint8)
    lab2 = cv2.merge((L, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def neutral_wb(img_bgr):
    """
    Mild gray-world (very conservative): prevents orange/blue casts,
    but avoids turning interiors “sterile”.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a = cv2.add(a, int((128 - np.mean(a)) * 0.35))
    b = cv2.add(b, int((128 - np.mean(b)) * 0.35))
    lab2 = cv2.merge((L, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def window_pull_blend(fused_bgr, under_bgr):
    """
    Subtle window recovery (no sticker look):
    blend underexposed only where fused highlights are too hot.
    """
    gray = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (gray > 220).astype(np.float32)  # window-ish highlights
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
    mask = np.clip(mask, 0.0, 1.0)

    alpha = (mask[..., None] * 0.70)  # subtle
    out = fused_bgr.astype(np.float32) * (1 - alpha) + under_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def crisp_sharpen(img_bgr):
    """
    Real estate sharp:
    unsharp mask tuned for downscaled images (looks “pro” not crunchy).
    """
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    out = cv2.addWeighted(img_bgr, 1.35, blur, -0.35, 0)
    return out


def estimate_vertical_rotation_deg(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=90, maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        v = abs(abs(ang) - 90)
        if v < 10:
            rotate = 90 - ang
            while rotate > 180:
                rotate -= 360
            while rotate < -180:
                rotate += 360
            if abs(rotate) <= 6:  # keep gentle
                angles.append(rotate)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image(img_bgr, deg):
    if abs(deg) < 0.1:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def process_once():
    print("Worker alive: checking for jobs...")

    job = pick_job()
    if not job:
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = job.get("attempts", 0)
    print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")

    try:
        rest_patch(
            "hdr_jobs",
            {"id": f"eq.{job_id}"},
            {"status": "processing", "locked_at": now_iso(), "attempts": attempts + 1},
        )

        s = get_set(set_id)
        order_id = s["order_id"]

        rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "processing"})
        rest_patch("hdr_orders", {"id": f"eq.{order_id}"}, {"status": "processing"})

        f_under = get_file(s["file_under_id"])
        f_base = get_file(s["file_base_id"])
        f_over = get_file(s["file_over_id"])

        print("Paths:", f_under["storage_path"], f_base["storage_path"], f_over["storage_path"])

        under = storage_download(RAW_BUCKET, f_under["storage_path"])
        base = storage_download(RAW_BUCKET, f_base["storage_path"])
        over = storage_download(RAW_BUCKET, f_over["storage_path"])

        # Auto-straighten (gentle)
        deg = estimate_vertical_rotation_deg(base)
        if abs(deg) >= 0.1:
            print("Auto-straighten deg:", deg)
            under = rotate_image(under, deg)
            base = rotate_image(base, deg)
            over = rotate_image(over, deg)

        # Merge
        fused = merge_mertens_clean(under, base, over)

        # Window pull (subtle)
        fused = window_pull_blend(fused, under)

        # Clean tone (bright/airy) + mild WB + crisp sharpen
        fused = tonal_clean_bright(fused)
        fused = neutral_wb(fused)
        fused = crisp_sharpen(fused)

        out_path = f"{order_id}/{set_id}.jpg"
        storage_upload(OUT_BUCKET, out_path, fused)

        rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "complete", "output_path": out_path})
        rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "complete"})

        # Mark order complete if all sets complete
        sets = rest_get("hdr_sets", {"select": "status", "order_id": f"eq.{order_id}"})
        if sets and all(x["status"] == "complete" for x in sets):
            rest_patch("hdr_orders", {"id": f"eq.{order_id}"}, {"status": "complete"})

        print("Completed set", set_id)
        return True

    except Exception as e:
        print("ERROR:", str(e))
        try:
            rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "error", "last_error": str(e)})
        except Exception:
            pass
        try:
            rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "error"})
        except Exception:
            pass
        return True


def main():
    print("HDR Worker started.")
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
