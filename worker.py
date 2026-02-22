import os
import time
import requests
import numpy as np
import cv2

# =========================
# CONFIG
# =========================
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

RAW_BUCKET = "hdr_raw"
OUT_BUCKET = "hdr_output"
POLL_SECONDS = 2

# Optional sky asset (upload your sky template into Supabase Storage)
SKY_BUCKET = os.environ.get("SKY_BUCKET", "").strip()          # e.g. "hdr_assets"
SKY_OBJECT = os.environ.get("SKY_OBJECT", "").strip()          # e.g. "skies/blue_sky.jpg"

# AI cleanup hook (OFF by default)
ENABLE_AI_CLEANUP = os.environ.get("ENABLE_AI_CLEANUP", "false").lower() == "true"
AI_CLEANUP_PROVIDER = os.environ.get("AI_CLEANUP_PROVIDER", "none").lower()
AI_CLEANUP_API_KEY = os.environ.get("AI_CLEANUP_API_KEY", "")

# Memory / quality caps (Railway-safe)
MAX_DIM = 1600  # upscale later if you move to bigger worker; 1600 is safe & looks good for MLS


# =========================
# HEADERS
# =========================
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


# =========================
# REST HELPERS
# =========================
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


# =========================
# DB HELPERS
# =========================
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


# =========================
# STORAGE
# =========================
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

    # immediate downscale for RAM safety
    img = downscale_max(img, MAX_DIM)
    return img


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


def try_load_sky_template():
    if not SKY_BUCKET or not SKY_OBJECT:
        return None
    try:
        sky = storage_download(SKY_BUCKET, SKY_OBJECT)
        return sky
    except Exception as e:
        print("Sky template load skipped:", str(e))
        return None


# =========================
# IMAGE UTILS
# =========================
def downscale_max(img, max_dim: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def gamma_correct(img_bgr, gamma=2.2):
    # apply gamma in a stable way
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def gray_world_wb(img_bgr, strength=0.6):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, int((128 - np.mean(a)) * strength))
    b = cv2.add(b, int((128 - np.mean(b)) * strength))
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_l_channel(img_bgr, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=grid)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def mild_vibrance(img_bgr, sat_mult=1.10):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(sat_mult)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gentle_sharpen(img_bgr, amount=0.20, sigma=1.0):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    return cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)


# =========================
# LUXURY FEATURES
# =========================
def estimate_vertical_rotation_deg(img_bgr):
    """
    Auto-straighten: detect dominant vertical line tilt and rotate.
    This is the single biggest perceived quality boost for interiors.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # convert to "verticalness": near 90 or -90
        # We care about lines that are close to vertical
        v = abs(abs(ang) - 90)
        if v < 12:  # within 12 degrees of vertical
            # how much rotate needed so line becomes perfectly vertical?
            # if ang=88, need +2 deg; if ang=92, need -2 deg
            rotate = 90 - ang
            # normalize rotate to [-15, 15]
            while rotate > 180:
                rotate -= 360
            while rotate < -180:
                rotate += 360
            if abs(rotate) <= 15:
                angles.append(rotate)

    if not angles:
        return 0.0

    # robust median
    return float(np.median(angles))


def rotate_image(img_bgr, deg):
    if abs(deg) < 0.1:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def is_exterior(img_bgr):
    """
    Lightweight heuristic: if the top band has lots of sky-like pixels, treat as exterior.
    """
    h, w = img_bgr.shape[:2]
    top = img_bgr[0 : max(1, int(h * 0.35)), :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)

    # sky-ish: blue/cyan hue, moderate sat, bright value
    lower = np.array([80, 30, 120], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return ratio > 0.10  # threshold tuned for typical exteriors


def sky_mask(img_bgr):
    """
    Create a mask for sky region.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 20, 110], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)

    # focus on upper half primarily
    h, w = img_bgr.shape[:2]
    m[int(h * 0.55):, :] = 0

    # refine
    m = cv2.medianBlur(m, 7)
    kernel = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    # feather edges for luxury blend
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    return m


def apply_sky_replacement(img_bgr, sky_bgr):
    if sky_bgr is None:
        return img_bgr

    m = sky_mask(img_bgr)  # 0..255-ish (blurred)
    if np.mean(m) < 3:
        return img_bgr

    h, w = img_bgr.shape[:2]
    sky = cv2.resize(sky_bgr, (w, h), interpolation=cv2.INTER_AREA)

    alpha = (m.astype(np.float32) / 255.0)[..., None]
    out = (img_bgr.astype(np.float32) * (1 - alpha) + sky.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_lawn(img_bgr):
    """
    Subtle luxury lawn pop: boost greens only, bottom area.
    """
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(h * 0.45):, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    # green hue range
    lower = np.array([35, 25, 40], dtype=np.float32)
    upper = np.array([95, 255, 255], dtype=np.float32)

    # mask greens
    hch = hsv[..., 0]
    sch = hsv[..., 1]
    vch = hsv[..., 2]

    mask = (hch >= lower[0]) & (hch <= upper[0]) & (sch >= lower[1]) & (vch >= lower[2])
    # subtle enhancements
    sch[mask] = np.clip(sch[mask] * 1.15, 0, 255)
    vch[mask] = np.clip(vch[mask] * 1.08, 0, 255)

    hsv[..., 1] = sch
    hsv[..., 2] = vch
    roi2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out = img_bgr.copy()
    out[int(h * 0.45):, :] = roi2
    return out


def window_pull_blend(fused_bgr, under_bgr):
    """
    Luxury window detail: use a highlight mask to blend underexposed where fused is too bright.
    Keeps it natural (no fake sticker look).
    """
    # luminance mask from fused
    gray = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # mask bright areas likely windows
    # threshold tuned for post-fusion brightness
    mask = (gray > 215).astype(np.float32)

    # feather / expand slightly
    mask = cv2.GaussianBlur(mask, (0, 0), 4.0)
    mask = np.clip(mask, 0.0, 1.0)

    # blend
    alpha = (mask[..., None] * 0.85)  # keep subtle
    out = fused_bgr.astype(np.float32) * (1 - alpha) + under_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def ai_object_removal_hook(img_bgr):
    """
    OPTIONAL: true luxury requires removing cords, tripods, reflections, etc.
    This step requires an external API. Off by default.
    You can wire ClipDrop Cleanup, Photoshop API, etc.
    """
    if not ENABLE_AI_CLEANUP or AI_CLEANUP_PROVIDER == "none":
        return img_bgr

    # Placeholder (kept safe): do nothing unless you implement a provider.
    # If you choose ClipDrop Cleanup, I’ll give you the exact code + endpoint.
    print("AI cleanup enabled but no provider implemented. Skipping.")
    return img_bgr


# =========================
# HDR + FINISHING
# =========================
def exposure_fusion_luxury(imgs_bgr):
    """
    Luxury-style: Natural fusion + normalize + gamma + finishing.
    """
    # Ensure same size
    h, w = imgs_bgr[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs_bgr]

    # Convert to float 0..1
    imgs_float = [(img.astype(np.float32) / 255.0) for img in imgs]

    mertens = cv2.createMergeMertens()
    fused = mertens.process(imgs_float)  # float32

    fused = np.clip(fused, 0.0, 1.0)

    # Normalize brightness (prevents dark/black)
    mx = float(fused.max())
    if mx > 1e-6:
        fused = fused / mx

    # Gamma for screen-friendly look
    fused = np.power(fused, 1.0 / 2.2)

    out = (fused * 255.0).clip(0, 255).astype(np.uint8)

    # Free
    del imgs_float
    del fused

    return out


def luxury_finish_pipeline(fused_bgr, under_bgr, base_bgr, over_bgr, sky_bgr=None):
    # 1) Vertical alignment (auto-straighten)
    deg = estimate_vertical_rotation_deg(fused_bgr)
    if abs(deg) >= 0.1:
        print("Auto-straighten deg:", deg)
        fused_bgr = rotate_image(fused_bgr, deg)
        under_bgr = rotate_image(under_bgr, deg)
        base_bgr = rotate_image(base_bgr, deg)
        over_bgr = rotate_image(over_bgr, deg)

    # 2) Window pull blending (underexposed in highlights)
    fused_bgr = window_pull_blend(fused_bgr, under_bgr)

    # 3) White balance + local contrast + vibrance + sharpen
    fused_bgr = gray_world_wb(fused_bgr, strength=0.55)
    fused_bgr = clahe_l_channel(fused_bgr, clip=2.0, grid=(8, 8))
    fused_bgr = mild_vibrance(fused_bgr, sat_mult=1.08)
    fused_bgr = gentle_sharpen(fused_bgr, amount=0.18, sigma=1.0)

    # 4) Exterior-only: sky replacement + lawn
    if is_exterior(fused_bgr):
        print("Exterior detected: applying sky + lawn")
        fused_bgr = apply_sky_replacement(fused_bgr, sky_bgr)
        fused_bgr = enhance_lawn(fused_bgr)

    # 5) Optional AI cleanup
    fused_bgr = ai_object_removal_hook(fused_bgr)

    return fused_bgr


# =========================
# ORDER CHECK
# =========================
def order_done(order_id: str):
    rows = rest_get("hdr_sets", {"select": "status", "order_id": f"eq.{order_id}"})
    return all(r["status"] == "complete" for r in rows)


# =========================
# PROCESS JOB
# =========================
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

        print("Paths:",
              f_under["storage_path"],
              f_base["storage_path"],
              f_over["storage_path"])

        # DOWNLOAD (already downscaled for safety)
        under = storage_download(RAW_BUCKET, f_under["storage_path"])
        base = storage_download(RAW_BUCKET, f_base["storage_path"])
        over = storage_download(RAW_BUCKET, f_over["storage_path"])

        # SKY TEMPLATE (optional)
        sky = try_load_sky_template()

        # FUSION
        fused = exposure_fusion_luxury([under, base, over])

        # FINISHING (luxury stack)
        final = luxury_finish_pipeline(fused, under, base, over, sky_bgr=sky)

        out_path = f"{order_id}/{set_id}.jpg"

        # UPLOAD
        storage_upload(OUT_BUCKET, out_path, final)

        # UPDATE DB
        rest_patch("hdr_sets", {"id": f"eq.{set_id}"}, {"status": "complete", "output_path": out_path})
        rest_patch("hdr_jobs", {"id": f"eq.{job_id}"}, {"status": "complete"})

        if order_done(order_id):
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


# =========================
# MAIN LOOP
# =========================
def main():
    print("HDR Worker started.")
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
