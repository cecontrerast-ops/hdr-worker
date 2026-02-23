import os
import time
import gc
import json
import math
import requests
import numpy as np
import cv2
from datetime import datetime, timezone

# =========================
# ENV / CONFIG
# =========================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.environ.get("OUT_BUCKET", "hdr_output")

# 512MB SAFE DEFAULTS
MAX_MERGE_DIM = int(os.environ.get("MAX_MERGE_DIM", "2000"))     # <= 2000 recommended for 0.5GB
OUTPUT_JPEG_QUALITY = int(os.environ.get("OUTPUT_JPEG_QUALITY", "92"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1.5"))
LOCK_SECONDS = int(os.environ.get("LOCK_SECONDS", "120"))

# LOOK TUNING (closer to AutoHDR)
WINDOW_THR = int(os.environ.get("WINDOW_THR", "210"))            # lower => stronger window pull
WINDOW_ALPHA = float(os.environ.get("WINDOW_ALPHA", "1.50"))     # 1.25–1.70 typical
HIGHLIGHT_ROLLOFF = float(os.environ.get("HIGHLIGHT_ROLLOFF", "0.30"))  # smaller => brighter highlights protection
SCURVE_BLEND = float(os.environ.get("SCURVE_BLEND", "0.60"))     # 0.55–0.70 => more pop
CLARITY = float(os.environ.get("CLARITY", "0.60"))               # 0.3–0.9
SHARP = float(os.environ.get("SHARP", "0.55"))                   # 0.3–0.8

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars.")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# =========================
# SUPABASE REST HELPERS
# =========================
def sb_get(path, params=None):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def sb_patch(path, body):
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()

def sb_post(path, body):
    url = f"{SUPABASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()

def patch_row(table, row_id, payload):
    return sb_patch(f"/rest/v1/{table}?id=eq.{row_id}", payload)

def get_row(table, row_id):
    rows = sb_get(f"/rest/v1/{table}", params={"id": f"eq.{row_id}", "select": "*"})
    return rows[0] if rows else None

def pick_job():
    # pick next queued, unlocked, oldest
    jobs = sb_get(
        "/rest/v1/hdr_jobs",
        params={
            "select": "*",
            "status": "eq.queued",
            "locked_at": "is.null",
            "order": "created_at.asc",
            "limit": "1",
        },
    )
    return jobs[0] if jobs else None

# =========================
# STORAGE HELPERS
# =========================
def storage_download(bucket, path):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.content
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {bucket}/{path}")
    return img

def storage_upload_jpg(bucket, path, bgr_img):
    # encode to jpg
    ok, enc = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.put(url, headers={**HEADERS, "Content-Type": "image/jpeg"}, data=enc.tobytes(), timeout=60)
    r.raise_for_status()

# =========================
# IMAGE PIPELINE (512MB SAFE)
# =========================
def downscale_to_max(img, max_dim):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def gray_world_wb(bgr):
    # cheap, good for mixed lighting
    b, g, r = cv2.split(bgr.astype(np.float32))
    avg_b = np.mean(b) + 1e-6
    avg_g = np.mean(g) + 1e-6
    avg_r = np.mean(r) + 1e-6
    avg = (avg_b + avg_g + avg_r) / 3.0
    b *= (avg / avg_b)
    g *= (avg / avg_g)
    r *= (avg / avg_r)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def highlight_rolloff(bgr, denom=0.30):
    # compress highlights slightly so walls stay clean but not blown
    x = bgr.astype(np.float32) / 255.0
    # soft knee
    y = x / (x + denom)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

def s_curve(bgr, blend=0.60):
    # adds punch without grungy HDR
    x = bgr.astype(np.float32) / 255.0
    # classic S curve
    y = 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))
    out = (blend * y + (1.0 - blend) * x)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def local_contrast_clarity(bgr, amount=0.6):
    if amount <= 0:
        return bgr
    # "clarity": unsharp-mask on luminance-like channel
    blur = cv2.GaussianBlur(bgr, (0, 0), 3.0)
    out = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return out

def sharpen(bgr, amount=0.55):
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
    out = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return out

def mertens_merge(bgr_list):
    # Mertens expects float32 0..1; keep only 3 images -> minimal stack
    h, w = bgr_list[0].shape[:2]
    resized = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in bgr_list]
    imgs_f = [im.astype(np.float32) / 255.0 for im in resized]
    mertens = cv2.createMergeMertens()
    fused = mertens.process(imgs_f)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out

def window_mask_blend(base_like, underexposed, thr=210, alpha=1.50):
    """
    AutoHDR look comes from: keep interior from merge/base,
    but pull windows from underexposed using a soft mask.
    """
    # mask from brightness of base_like (detect blown areas)
    gray = cv2.cvtColor(base_like, cv2.COLOR_BGR2GRAY)
    # soft mask: 0..1
    m = (gray.astype(np.float32) - thr) / (255.0 - thr + 1e-6)
    m = np.clip(m, 0.0, 1.0)
    # soften edges (no sticker look)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    m = np.clip(m * alpha, 0.0, 1.0)

    m3 = np.dstack([m, m, m])
    out = (base_like.astype(np.float32) * (1.0 - m3) + underexposed.astype(np.float32) * (m3)).astype(np.uint8)
    return out

def auto_vertical_soft(bgr):
    """
    512MB-safe *light* vertical correction:
    - finds a dominant vertical direction from edges
    - applies a small rotate to reduce tilt
    (Not full keystone, but improves perceived verticals)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 140)
    if lines is None:
        return bgr

    # collect angles near vertical
    angles = []
    for rho_theta in lines[:120]:
        rho, theta = rho_theta[0]
        # theta is angle of normal; line angle = theta - 90deg
        line_angle = (theta - np.pi / 2.0) * 180.0 / np.pi
        # keep near vertical lines (around 0 deg)
        if abs(line_angle) < 15:
            angles.append(line_angle)

    if not angles:
        return bgr

    # rotate by negative median
    rot = float(np.median(angles))
    if abs(rot) < 0.3:
        return bgr

    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, 1.0)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out

def process_hdr_triplet(img_under, img_base, img_over):
    # 1) downscale early to stay in RAM
    img_under = downscale_to_max(img_under, MAX_MERGE_DIM)
    img_base  = downscale_to_max(img_base,  MAX_MERGE_DIM)
    img_over  = downscale_to_max(img_over,  MAX_MERGE_DIM)

    # 2) match sizes
    h, w = img_base.shape[:2]
    img_under = cv2.resize(img_under, (w, h), interpolation=cv2.INTER_AREA)
    img_over  = cv2.resize(img_over,  (w, h), interpolation=cv2.INTER_AREA)

    # 3) Mertens merge (cheap HDR base)
    merged = mertens_merge([img_under, img_base, img_over])

    # free quickly
    del img_over
    gc.collect()

    # 4) “AutoHDR look” stack (cheap steps)
    merged = gray_world_wb(merged)
    merged = highlight_rolloff(merged, denom=HIGHLIGHT_ROLLOFF)
    merged = s_curve(merged, blend=SCURVE_BLEND)

    # 5) window pull (key difference)
    merged = window_mask_blend(merged, img_under, thr=WINDOW_THR, alpha=WINDOW_ALPHA)

    # 6) clarity + sharpen (safe)
    merged = local_contrast_clarity(merged, amount=CLARITY)
    merged = sharpen(merged, amount=SHARP)

    # 7) soft vertical correction (not keystone, but helps)
    merged = auto_vertical_soft(merged)

    return merged

# =========================
# JOB / ORDER PROCESSOR
# =========================
def get_file(file_id):
    return get_row("hdr_files", file_id)

def get_set(set_id):
    return get_row("hdr_sets", set_id)

def order_done(order_id):
    sets = sb_get("/rest/v1/hdr_sets", params={"order_id": f"eq.{order_id}", "select": "status"})
    return all(s.get("status") == "complete" for s in sets) if sets else False

def process_once():
    job = pick_job()
    if not job:
        print("Worker alive: checking for jobs...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = int(job.get("attempts", 0))

    # lock job
    lock_time = now_iso()
    patch_row("hdr_jobs", job_id, {"locked_at": lock_time, "status": "processing", "attempts": attempts + 1})

    try:
        s = get_set(set_id)
        if not s:
            raise RuntimeError(f"Set not found: {set_id}")

        order_id = s["order_id"]
        patch_row("hdr_sets", set_id, {"status": "processing"})
        patch_row("hdr_orders", order_id, {"status": "processing"})

        f_under = get_file(s["file_under_id"])
        f_base  = get_file(s["file_base_id"])
        f_over  = get_file(s["file_over_id"])
        if not f_under or not f_base or not f_over:
            raise RuntimeError("Missing file rows for set")

        p_under = f_under["storage_path"]
        p_base  = f_base["storage_path"]
        p_over  = f_over["storage_path"]

        print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")
        print(f"Paths: {p_under} {p_base} {p_over}")

        print(f"Downloading: {RAW_BUCKET}/{p_under}")
        img_under = storage_download(RAW_BUCKET, p_under)
        print(f"Downloading: {RAW_BUCKET}/{p_base}")
        img_base = storage_download(RAW_BUCKET, p_base)
        print(f"Downloading: {RAW_BUCKET}/{p_over}")
        img_over = storage_download(RAW_BUCKET, p_over)

        out = process_hdr_triplet(img_under, img_base, img_over)

        # free big inputs now
        del img_under, img_base, img_over
        gc.collect()

        out_path = f"{order_id}/{set_id}.jpg"
        print(f"Uploading: {OUT_BUCKET}/{out_path}")
        storage_upload_jpg(OUT_BUCKET, out_path, out)

        del out
        gc.collect()

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete"})

        if order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})

        print(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = str(e)
        print("ERROR:", err)
        patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
        try:
            patch_row("hdr_sets", set_id, {"status": "error"})
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
