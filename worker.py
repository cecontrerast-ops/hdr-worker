import os
import time
import json
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
import cv2

# =========================
# ENV / CONFIG
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

RAW_BUCKET_ENV = os.getenv("RAW_BUCKET", "hdr_raw")
OUT_BUCKET_ENV = os.getenv("OUT_BUCKET", "hdr_output")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "6"))

# Quality knobs (AutoHDR-ish)
MAX_SIDE = int(os.getenv("MAX_SIDE", "2600"))  # downscale safety
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))

# Brightness & contrast tuning
TARGET_MID = float(os.getenv("TARGET_MID", "0.62"))   # higher = brighter (0.58-0.70)
BLACK_LIFT = float(os.getenv("BLACK_LIFT", "0.03"))   # lift shadows a bit
SAT_BOOST = float(os.getenv("SAT_BOOST", "1.05"))     # subtle

# Window pull tuning
WINDOW_V_THRESH = int(os.getenv("WINDOW_V_THRESH", "210"))  # lower pulls more window detail
WINDOW_ALPHA_GAIN = float(os.getenv("WINDOW_ALPHA_GAIN", "1.5"))

# Sharpness / clarity
UNSHARP_AMOUNT = float(os.getenv("UNSHARP_AMOUNT", "0.6"))  # 0.3-0.9
UNSHARP_RADIUS = float(os.getenv("UNSHARP_RADIUS", "1.2"))

# Vertical correction strength (keep mild)
VERTICAL_FIX = os.getenv("VERTICAL_FIX", "1") == "1"  # can disable
VERTICAL_MAX_TILT_DEG = float(os.getenv("VERTICAL_MAX_TILT_DEG", "3.0"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =========================
# HTTP / SUPABASE HELPERS
# =========================

def _safe_json(resp: requests.Response):
    """
    Supabase/PostgREST sometimes returns empty body (204) or non-JSON on errors.
    This prevents JSONDecodeError.
    """
    if resp.status_code == 204:
        return None
    txt = (resp.text or "").strip()
    if not txt:
        return None
    try:
        return resp.json()
    except Exception:
        return {"_non_json": True, "status": resp.status_code, "text": txt[:500]}

def sb_get(path: str):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return _safe_json(r)

def sb_post(path: str, payload: dict):
    url = f"{SUPABASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return _safe_json(r)

def sb_patch(path: str, payload: dict):
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return _safe_json(r)

def patch_row(table: str, row_id: str, payload: dict):
    # PostgREST patch by id
    return sb_patch(f"/rest/v1/{table}?id=eq.{row_id}", payload)

def get_one(table: str, row_id: str):
    rows = sb_get(f"/rest/v1/{table}?id=eq.{row_id}&select=*")
    if not rows:
        return None
    return rows[0]


# =========================
# STORAGE HELPERS
# =========================

def list_buckets():
    # Storage buckets endpoint
    url = f"{SUPABASE_URL}/storage/v1/bucket"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = _safe_json(r) or []
    return [b.get("name") for b in data if isinstance(b, dict) and b.get("name")]

def resolve_buckets():
    """
    Auto-resolve bucket names if env vars are wrong / or mismatch.
    """
    buckets = list_buckets()
    resolved_raw = RAW_BUCKET_ENV if RAW_BUCKET_ENV in buckets else None
    resolved_out = OUT_BUCKET_ENV if OUT_BUCKET_ENV in buckets else None

    # Fallbacks (common names)
    for cand in ["hdr_raw", "raw", "uploads", "hdr-raw"]:
        if not resolved_raw and cand in buckets:
            resolved_raw = cand
    for cand in ["hdr_output", "output", "exports", "hdr-out", "hdr_results"]:
        if not resolved_out and cand in buckets:
            resolved_out = cand

    if not resolved_raw or not resolved_out:
        raise RuntimeError(f"Could not resolve buckets. Visible: {buckets}")

    return resolved_raw, resolved_out, buckets

RAW_BUCKET, OUT_BUCKET, VISIBLE_BUCKETS = resolve_buckets()
print("=== HDR WORKER v6 (BLACK FIX + AUTOHDR FINISH) LOADED ===")
print("SUPABASE_URL =", SUPABASE_URL)
print("RAW_BUCKET (env) =", RAW_BUCKET_ENV)
print("OUT_BUCKET (env) =", OUT_BUCKET_ENV)
print("Buckets visible to service role:", VISIBLE_BUCKETS)
print("Resolved_RAW_BUCKET =", RAW_BUCKET)
print("Resolved_OUT_BUCKET =", OUT_BUCKET)

def storage_download(bucket: str, path: str) -> bytes:
    # NOTE: path is object path, no leading slash
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Download failed {bucket}/{path}: {r.status_code} {r.text[:300]}")
    return r.content

def storage_upload_jpg(bucket: str, path: str, img_bgr_uint8: np.ndarray):
    if img_bgr_uint8 is None:
        raise RuntimeError("storage_upload_jpg: image is None")

    if img_bgr_uint8.dtype != np.uint8:
        img_bgr_uint8 = np.clip(img_bgr_uint8, 0, 255).astype(np.uint8)

    if len(img_bgr_uint8.shape) == 2:
        img_bgr_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_GRAY2BGR)

    ok, buf = cv2.imencode(".jpg", img_bgr_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for jpg")

    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = dict(HEADERS)
    headers["Content-Type"] = "image/jpeg"
    r = requests.put(url, headers=headers, data=buf.tobytes(), timeout=120)
    if not r.ok:
        raise RuntimeError(f"Upload failed {bucket}/{path}: {r.status_code} {r.text[:300]}")


# =========================
# IMAGE PIPELINE (AUTOHDR-LIKE)
# =========================

def downscale_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def align_to_base(img: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Fast-ish alignment using ECC on grayscale.
    Helps avoid softness/ghosting.
    """
    try:
        im1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        warp = np.eye(2, 3, dtype=np.float32)  # affine
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        cc, warp = cv2.findTransformECC(im1, im2, warp, cv2.MOTION_AFFINE, criteria, None, 1)

        aligned = cv2.warpAffine(img, warp, (base.shape[1], base.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception:
        return img  # safe fallback

def exposure_fusion_mertens(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    """
    Returns float32 image in 0..1 range (approx).
    """
    m = cv2.createMergeMertens()
    imgs_f = [np.clip(im.astype(np.float32) / 255.0, 0.0, 1.0) for im in imgs_bgr]
    fused = m.process(imgs_f)  # float32 HxWx3
    return fused

def normalize_exposure_u8(img_float01: np.ndarray) -> np.ndarray:
    """
    BLACK FIX:
    - convert to uint8 safely
    - auto-normalize midtones so image never comes out near-black
    """
    x = np.nan_to_num(img_float01, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)

    # Compute perceived luminance
    lum = 0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]
    p = np.percentile(lum, [1, 50, 99]).astype(np.float32)
    p1, pmid, p99 = float(p[0]), float(p[1]), float(p[2])

    # If the mid is too low -> lift globally
    if pmid < 1e-4:
        gain = 6.0
    else:
        gain = TARGET_MID / pmid

    # Prevent insane gain
    gain = float(np.clip(gain, 0.6, 3.5))

    x = x * gain

    # Soft highlight rolloff to avoid washed windows
    x = x / (x + 0.30)  # smaller = brighter overall, but controlled

    # Lift shadows slightly
    x = np.clip(x + BLACK_LIFT, 0.0, 1.0)

    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return out

def window_mask_blend(fused_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    """
    Simple, strong window pull:
    - detect bright window regions in fused image
    - blend in the UNDER exposure in those areas
    """
    try:
        fused = fused_bgr.copy()
        under = cv2.resize(under_bgr, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(np.float32)

        # Window-ish mask: very bright areas
        mask = (v > float(WINDOW_V_THRESH)).astype(np.float32)

        # Smooth / feather
        mask = cv2.GaussianBlur(mask, (0, 0), 6)
        mask = np.clip(mask * WINDOW_ALPHA_GAIN, 0.0, 1.0)

        # Blend under exposure where windows are bright
        out = fused.astype(np.float32) * (1.0 - mask[..., None]) + under.astype(np.float32) * (mask[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return fused_bgr

def boost_floor_microcontrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Adds subtle clarity mostly to darker midtones (often floors).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    Lf = L.astype(np.float32) / 255.0
    # local contrast
    blur = cv2.GaussianBlur(Lf, (0, 0), 2.0)
    detail = Lf - blur
    # emphasize mid/dark regions a bit
    w = np.clip((0.65 - Lf) * 1.3, 0.0, 0.6)
    L2 = np.clip(Lf + detail * (0.8 * w), 0.0, 1.0)

    L_out = (L2 * 255.0).astype(np.uint8)
    lab2 = cv2.merge([L_out, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def unsharp(img_bgr: np.ndarray) -> np.ndarray:
    if UNSHARP_AMOUNT <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), UNSHARP_RADIUS)
    out = cv2.addWeighted(img_bgr, 1.0 + float(UNSHARP_AMOUNT), blur, -float(UNSHARP_AMOUNT), 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def sat_boost(img_bgr: np.ndarray) -> np.ndarray:
    if abs(SAT_BOOST - 1.0) < 1e-3:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(SAT_BOOST)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def vertical_correction(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mild correction: tries to keep verticals straighter.
    This is intentionally conservative (no crazy warps).
    """
    if not VERTICAL_FIX:
        return img_bgr

    # Detect edges and dominant angle (very light heuristic)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=200, maxLineGap=20)
    if lines is None:
        return img_bgr

    # Estimate skew from near-vertical lines
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dy) < 1:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # normalize to [0..180)
        ang = (ang + 180.0) % 180.0
        # vertical lines near 90
        if 75.0 <= ang <= 105.0:
            angles.append(ang - 90.0)  # skew from vertical

    if len(angles) < 6:
        return img_bgr

    skew = float(np.median(angles))  # degrees
    skew = float(np.clip(skew, -VERTICAL_MAX_TILT_DEG, VERTICAL_MAX_TILT_DEG))
    if abs(skew) < 0.2:
        return img_bgr

    # Rotate opposite to skew (small)
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), skew, 1.0)
    out = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out

def make_hdr_like_autohdr(img_under: np.ndarray, img_base: np.ndarray, img_over: np.ndarray) -> np.ndarray:
    """
    Full pipeline:
    - Downscale
    - Align
    - Mertens fusion
    - Normalize exposure (BLACK FIX)
    - Window pull
    - Floor pop
    - Vertical correction
    - Saturation + Sharpness
    """
    # downscale for safety + speed
    under = downscale_max_side(img_under, MAX_SIDE)
    base  = downscale_max_side(img_base,  MAX_SIDE)
    over  = downscale_max_side(img_over,  MAX_SIDE)

    # match sizes to base
    h, w = base.shape[:2]
    under = cv2.resize(under, (w, h), interpolation=cv2.INTER_AREA)
    over  = cv2.resize(over,  (w, h), interpolation=cv2.INTER_AREA)

    # align brackets to base (reduces softness / ghosting)
    under_a = align_to_base(under, base)
    over_a  = align_to_base(over,  base)

    fused_f = exposure_fusion_mertens([under_a, base, over_a])   # float 0..1
    fused_u8 = normalize_exposure_u8(fused_f)                    # BLACK FIX (auto midtone)

    # window pull using under exposure
    fused_u8 = window_mask_blend(fused_u8, under_a)

    # floor clarity
    fused_u8 = boost_floor_microcontrast(fused_u8)

    # vertical correction (mild)
    fused_u8 = vertical_correction(fused_u8)

    # subtle color + sharpness
    fused_u8 = sat_boost(fused_u8)
    fused_u8 = unsharp(fused_u8)

    return fused_u8


# =========================
# JOB / WORKER LOOP
# =========================

def pick_next_job():
    """
    Picks the oldest queued job that is not locked or lock expired.
    You likely already have this logic in DB; we keep it simple.
    """
    # Get queued jobs
    jobs = sb_get("/rest/v1/hdr_jobs?status=eq.queued&select=*&order=created_at.asc&limit=10") or []
    now = datetime.now(timezone.utc)

    for j in jobs:
        job_id = j["id"]
        attempts = int(j.get("attempts") or 0)
        locked_at = j.get("locked_at")

        if attempts >= MAX_ATTEMPTS:
            patch_row("hdr_jobs", job_id, {"status": "error", "last_error": "Max attempts reached"})
            continue

        # If locked_at exists, skip (simple). If you want lock expiry, implement it here.
        if locked_at:
            continue

        # lock it
        patch_row("hdr_jobs", job_id, {"status": "processing", "locked_at": now_iso(), "attempts": attempts + 1})
        return j

    return None

def download_image(storage_path: str) -> np.ndarray:
    data = storage_download(RAW_BUCKET, storage_path)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imdecode failed for {storage_path}")
    return img

def process_once() -> bool:
    job = pick_next_job()
    if not job:
        print("Worker alive: checking for jobs...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    attempts = int(job.get("attempts") or 0)

    try:
        s = get_one("hdr_sets", set_id)
        if not s:
            patch_row("hdr_jobs", job_id, {"status": "error", "last_error": f"Missing hdr_sets row {set_id}"})
            return True

        order_id = s.get("order_id")
        patch_row("hdr_sets", set_id, {"status": "processing"})
        if order_id:
            patch_row("hdr_orders", order_id, {"status": "processing"})

        f_under = get_one("hdr_files", s["file_under_id"])
        f_base  = get_one("hdr_files", s["file_base_id"])
        f_over  = get_one("hdr_files", s["file_over_id"])

        if not f_under or not f_base or not f_over:
            raise RuntimeError("Missing hdr_files rows for set")

        p_under = f_under["storage_path"]
        p_base  = f_base["storage_path"]
        p_over  = f_over["storage_path"]

        print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")
        print(f"Paths: {p_under} {p_base} {p_over}")

        img_under = download_image(p_under)
        img_base  = download_image(p_base)
        img_over  = download_image(p_over)

        out = make_hdr_like_autohdr(img_under, img_base, img_over)

        out_path = f"{order_id}/{set_id}.jpg" if order_id else f"{set_id}.jpg"
        print(f"Uploading: {OUT_BUCKET}/{out_path}")
        storage_upload_jpg(OUT_BUCKET, out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete"})

        # If you have "order_done" logic, keep yours; this is safe fallback:
        if order_id:
            # mark complete when all sets are complete
            sets = sb_get(f"/rest/v1/hdr_sets?order_id=eq.{order_id}&select=status") or []
            if sets and all(x.get("status") == "complete" for x in sets):
                patch_row("hdr_orders", order_id, {"status": "complete"})

        print(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        print("ERROR:", err)
        print(traceback.format_exc())
        patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
        try:
            patch_row("hdr_sets", set_id, {"status": "error"})
        except Exception:
            pass
        return True

def main():
    while True:
        did = process_once()
        if not did:
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
