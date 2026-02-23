import os
import time
import json
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
import cv2


# =========================
# ENV
# =========================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

RAW_BUCKET = os.environ.get("RAW_BUCKET", "hdr_raw").strip()
OUT_BUCKET = os.environ.get("OUT_BUCKET", "hdr_output").strip()

POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))

# Quality controls (tune if needed)
MAX_MERGE_DIM = int(os.environ.get("MAX_MERGE_DIM", "2600"))  # merge at up to ~2600px longest side
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

# Vertical correction controls (safe defaults)
ENABLE_VERTICAL_CORRECTION = os.environ.get("ENABLE_VERTICAL_CORRECTION", "true").lower() == "true"
MAX_ROLL_DEG = float(os.environ.get("MAX_ROLL_DEG", "3.0"))  # clamp roll correction to avoid artifacts

# Floor pop controls
ENABLE_FLOOR_POP = os.environ.get("ENABLE_FLOOR_POP", "true").lower() == "true"

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars.")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

REST = f"{SUPABASE_URL}/rest/v1"
STORAGE = f"{SUPABASE_URL}/storage/v1"


# =========================
# HELPERS: TIME/LOG
# =========================
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str):
    print(msg, flush=True)


# =========================
# SUPABASE REST HELPERS
# =========================
def rest_get(table: str, params: str):
    url = f"{REST}/{table}?{params}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def rest_patch(table: str, match_params: str, payload: dict):
    url = f"{REST}/{table}?{match_params}"
    r = requests.patch(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json() if r.text else None


def patch_row(table: str, row_id: str, payload: dict):
    return rest_patch(table, f"id=eq.{row_id}", payload)


def get_one(table: str, row_id: str):
    rows = rest_get(table, f"id=eq.{row_id}&limit=1")
    return rows[0] if rows else None


# =========================
# STORAGE HELPERS
# =========================
def storage_download(bucket: str, path: str) -> np.ndarray:
    url = f"{STORAGE}/object/{bucket}/{path}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}", "apikey": SUPABASE_SERVICE_ROLE_KEY},
        timeout=90,
    )
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {bucket}/{path}")
    return img


def storage_upload_jpg(bucket: str, path: str, img_bgr: np.ndarray):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError("Failed to encode JPG")

    url = f"{STORAGE}/object/{bucket}/{path}"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "image/jpeg",
        },
        data=buf.tobytes(),
        timeout=120,
    )
    if r.status_code == 409:
        r = requests.put(
            url,
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Content-Type": "image/jpeg",
            },
            data=buf.tobytes(),
            timeout=120,
        )
    r.raise_for_status()


# =========================
# IMAGE HELPERS
# =========================
def resize_to_max(img: np.ndarray, max_dim: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    scale = max_dim / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def align_ecc(base_bgr: np.ndarray, target_bgr: np.ndarray):
    base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    base_s, _ = resize_to_max(base, 1400)
    tgt_s, _ = resize_to_max(tgt, 1400)

    h, w = base_s.shape[:2]
    tgt_s = cv2.resize(tgt_s, (w, h), interpolation=cv2.INTER_AREA)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6)

    try:
        _, warp = cv2.findTransformECC(base_s, tgt_s, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except Exception:
        return target_bgr

    H, W = base_bgr.shape[:2]
    aligned = cv2.warpAffine(
        target_bgr,
        warp,
        (W, H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def merge_mertens(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    m = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=0.6, exposure_weight=0.8)
    imgs_f = [img.astype(np.float32) / 255.0 for img in imgs_bgr]
    fused = m.process(imgs_f)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out


def gray_world_wb(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# Brighter overall + better highlight control
# -------------------------
def highlight_rolloff(img_bgr: np.ndarray) -> np.ndarray:
    """
    Soft highlight compression, tuned brighter overall.
    (Changed denominator 0.35 -> 0.30)
    """
    x = img_bgr.astype(np.float32) / 255.0
    y = x / (x + 0.30)
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return out


def local_contrast(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def s_curve(img_bgr: np.ndarray) -> np.ndarray:
    """
    Brighter / more premium midtone lift
    (Changed blend 0.55/0.45 -> 0.45/0.55)
    """
    x = img_bgr.astype(np.float32) / 255.0
    y = 1 / (1 + np.exp(-8 * (x - 0.5)))
    y = 0.45 * x + 0.55 * y
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return out


# -------------------------
# Stronger window pulls
# -------------------------
def window_mask_blend(fused_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # Stronger: (v > 220) -> (v > 210)
    mask = ((v > 210) & (s < 80)).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=6)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    # Stronger: 1.25 -> 1.5
    alpha = np.clip(alpha * 1.5, 0, 1)

    fused_f = fused_bgr.astype(np.float32)
    under_f = under_bgr.astype(np.float32)
    out = fused_f * (1 - alpha) + under_f * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_aware_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.1, sigmaY=1.1)
    sharp = cv2.addWeighted(img_bgr, 1.35, blur, -0.35, 0)
    sharp = cv2.detailEnhance(sharp, sigma_s=10, sigma_r=0.15)
    return sharp


# =========================
# VERTICAL CORRECTION (ROLL)
# =========================
def estimate_roll_degrees(img_bgr: np.ndarray) -> float:
    """
    Estimate roll angle using Hough lines.
    We use near-vertical lines and compute their deviation from vertical.
    Returns degrees to rotate (positive = rotate CCW).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray, _ = resize_to_max(gray, 1400)

    # edges
    edges = cv2.Canny(gray, 60, 160, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=15)
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        # angle from horizontal
        ang = np.degrees(np.arctan2(dy, dx))
        # convert to deviation from vertical:
        # vertical lines ~ +/-90
        # deviation = ang - 90 (or ang + 90), choose closer
        dev1 = ang - 90.0
        dev2 = ang + 90.0
        dev = dev1 if abs(dev1) < abs(dev2) else dev2

        # keep only lines that are close to vertical (within 25 deg)
        if abs(dev) <= 25:
            # weight longer lines more
            length = math.hypot(dx, dy)
            angles.append((dev, length))

    if not angles:
        return 0.0

    # weighted median-ish: sort by dev, weight by length
    angles.sort(key=lambda t: t[0])
    devs = np.array([a[0] for a in angles], dtype=np.float32)
    wts = np.array([a[1] for a in angles], dtype=np.float32)
    wts = wts / (np.sum(wts) + 1e-6)

    # weighted average is stable for our case
    dev = float(np.sum(devs * wts))
    # rotate opposite of deviation to make vertical
    return -dev


def rotate_keep_bounds(img_bgr: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 0.05:
        return img_bgr

    h, w = img_bgr.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(img_bgr, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # crop back to original aspect center
    if nW > w and nH > h:
        x0 = (nW - w) // 2
        y0 = (nH - h) // 2
        rotated = rotated[y0:y0 + h, x0:x0 + w]
    else:
        rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_CUBIC)
    return rotated


def auto_vertical_correction(img_bgr: np.ndarray) -> np.ndarray:
    """
    Correct roll (tilt). Clamp to avoid over-correction.
    """
    deg = estimate_roll_degrees(img_bgr)
    if abs(deg) > MAX_ROLL_DEG:
        deg = float(np.sign(deg) * MAX_ROLL_DEG)
    if abs(deg) < 0.05:
        return img_bgr
    return rotate_keep_bounds(img_bgr, deg)


# =========================
# FLOOR POP
# =========================
def floor_pop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Subtle premium floor pop:
    - create a mask in lower portion where texture is high and luminance is mid/dark
    - apply micro-contrast + slight lift + detail on mask only
    """
    h, w = img_bgr.shape[:2]

    # work on a downscaled copy for mask creation
    small, scale = resize_to_max(img_bgr, 1400)
    hs, ws = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # texture (edges)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # base region = lower 55%
    region = np.zeros((hs, ws), dtype=np.uint8)
    region[int(hs * 0.45):, :] = 255

    # floor heuristic:
    # - not too bright (avoid walls/windows)
    # - has texture
    # - moderate saturation OK
    mask = (
        (region == 255) &
        (v > 40) & (v < 200) &
        (mag > 25)
    ).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 7)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 7, 7)

    # upscale mask to full res
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    alpha = (mask_full.astype(np.float32) / 255.0)[..., None]
    alpha = np.clip(alpha * 0.9, 0, 1)  # subtle

    # make a "popped" version
    popped = img_bgr.copy()

    # micro contrast on L channel (stronger tile/wood clarity)
    lab = cv2.cvtColor(popped, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    popped = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # slight lift
    popped = cv2.convertScaleAbs(popped, alpha=1.04, beta=6)

    # detail
    popped = cv2.detailEnhance(popped, sigma_s=12, sigma_r=0.18)

    # blend only in floor mask
    out = img_bgr.astype(np.float32) * (1 - alpha) + popped.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# MAIN HDR PIPELINE (AUTOHDR-LIKE++)
# =========================
def autohdr_like_pipeline(under_bgr: np.ndarray, base_bgr: np.ndarray, over_bgr: np.ndarray) -> np.ndarray:
    # Ensure same size
    H, W = base_bgr.shape[:2]
    under_bgr = cv2.resize(under_bgr, (W, H), interpolation=cv2.INTER_AREA)
    over_bgr = cv2.resize(over_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # Align
    under_a = align_ecc(base_bgr, under_bgr)
    over_a = align_ecc(base_bgr, over_bgr)

    # Merge at safe size
    base_s, _ = resize_to_max(base_bgr, MAX_MERGE_DIM)
    hs, ws = base_s.shape[:2]
    under_s = cv2.resize(under_a, (ws, hs), interpolation=cv2.INTER_AREA)
    over_s = cv2.resize(over_a, (ws, hs), interpolation=cv2.INTER_AREA)

    fused_s = merge_mertens([under_s, base_s, over_s])

    # Clean / premium look
    fused_s = gray_world_wb(fused_s)
    fused_s = highlight_rolloff(fused_s)
    fused_s = local_contrast(fused_s)
    fused_s = s_curve(fused_s)

    # Window pull
    fused_s = window_mask_blend(fused_s, under_s)

    # Back to full res
    fused = cv2.resize(fused_s, (W, H), interpolation=cv2.INTER_CUBIC)

    # Vertical correction (roll)
    if ENABLE_VERTICAL_CORRECTION:
        fused = auto_vertical_correction(fused)

    # Floor pop
    if ENABLE_FLOOR_POP:
        fused = floor_pop(fused)

    # Final crispness
    fused = edge_aware_sharpen(fused)

    return fused


# =========================
# JOB LOGIC
# =========================
def fetch_next_job():
    rows = rest_get("hdr_jobs", "status=eq.queued&locked_at=is.null&order=created_at.asc&limit=1")
    return rows[0] if rows else None


def order_done(order_id: str):
    sets = rest_get("hdr_sets", f"order_id=eq.{order_id}&limit=200")
    for s in sets:
        if s.get("status") not in ("complete",):
            return False
    return True


def process_once():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        log("Worker alive but missing SUPABASE env vars.")
        return False

    job = fetch_next_job()
    if not job:
        log("Worker alive: checking for jobs...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = int(job.get("attempts") or 0)

    log(f"Picked job {job_id} set_id={set_id} attempts={attempts}")

    patch_row("hdr_jobs", job_id, {
        "status": "processing",
        "locked_at": now_iso(),
        "attempts": attempts + 1
    })

    try:
        s = get_one("hdr_sets", set_id)
        if not s:
            raise RuntimeError(f"hdr_sets not found: {set_id}")

        order_id = s["order_id"]
        patch_row("hdr_sets", set_id, {"status": "processing"})
        patch_row("hdr_orders", order_id, {"status": "processing"})

        f_under = get_one("hdr_files", s["file_under_id"])
        f_base = get_one("hdr_files", s["file_base_id"])
        f_over = get_one("hdr_files", s["file_over_id"])
        if not f_under or not f_base or not f_over:
            raise RuntimeError("Missing hdr_files rows for set")

        p_under = f_under["storage_path"]
        p_base = f_base["storage_path"]
        p_over = f_over["storage_path"]

        log(f"Paths: {p_under} {p_base} {p_over}")

        log(f"Downloading: {RAW_BUCKET}/{p_under}")
        img_under = storage_download(RAW_BUCKET, p_under)
        log(f"Downloading: {RAW_BUCKET}/{p_base}")
        img_base = storage_download(RAW_BUCKET, p_base)
        log(f"Downloading: {RAW_BUCKET}/{p_over}")
        img_over = storage_download(RAW_BUCKET, p_over)

        out = autohdr_like_pipeline(img_under, img_base, img_over)

        out_path = f"{order_id}/{set_id}.jpg"
        log(f"Uploading: {OUT_BUCKET}/{out_path}")
        storage_upload_jpg(OUT_BUCKET, out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete", "last_error": None})

        if order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})

        log(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        log("ERROR processing job:\n" + err)
        log(traceback.format_exc())

        try:
            if attempts + 1 >= MAX_ATTEMPTS:
                patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
                try:
                    patch_row("hdr_sets", set_id, {"status": "error"})
                except Exception:
                    pass
            else:
                patch_row("hdr_jobs", job_id, {"status": "queued", "locked_at": None, "last_error": err})
                try:
                    patch_row("hdr_sets", set_id, {"status": "queued"})
                except Exception:
                    pass
        except Exception:
            pass

        return True


def main():
    log("HDR Worker started.")
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
