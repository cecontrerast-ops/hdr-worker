import os
import time
import json
import math
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

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars.")
    # We don't exit hard to keep container alive for debugging, but worker won't do anything.

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


def rest_post(table: str, payload: dict):
    url = f"{REST}/{table}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json() if r.text else None


def patch_row(table: str, row_id: str, payload: dict):
    # assumes tables use 'id' uuid
    return rest_patch(table, f"id=eq.{row_id}", payload)


def get_one(table: str, row_id: str):
    rows = rest_get(table, f"id=eq.{row_id}&limit=1")
    return rows[0] if rows else None


# =========================
# STORAGE HELPERS
# =========================
def storage_download(bucket: str, path: str) -> np.ndarray:
    # downloads bytes, decodes jpg/png with OpenCV
    url = f"{STORAGE}/object/{bucket}/{path}"
    r = requests.get(url, headers={"Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}", "apikey": SUPABASE_SERVICE_ROLE_KEY}, timeout=60)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {bucket}/{path}")
    return img


def storage_upload_jpg(bucket: str, path: str, img_bgr: np.ndarray):
    # encode JPG
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
    # If object exists, Supabase Storage returns 409; in that case do PUT
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
# IMAGE PIPELINE (AUTOHDR-LIKE)
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
    """
    Align target to base using ECC (translation+rotation).
    This helps ghosting and improves window detail.
    """
    base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    # Downscale for speed
    base_s, s1 = resize_to_max(base, 1400)
    tgt_s, s2 = resize_to_max(tgt, 1400)

    # Keep same size
    h, w = base_s.shape[:2]
    tgt_s = cv2.resize(tgt_s, (w, h), interpolation=cv2.INTER_AREA)

    # Motion model: Euclidean (rotation + translation)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6)

    try:
        cc, warp = cv2.findTransformECC(base_s, tgt_s, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except Exception:
        # Fallback: no alignment
        return target_bgr

    # Apply warp at full res
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
    """
    Exposure fusion (natural) – base layer.
    """
    m = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=0.6, exposure_weight=0.8)
    imgs_f = [img.astype(np.float32) / 255.0 for img in imgs_bgr]
    fused = m.process(imgs_f)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out


def gray_world_wb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple white balance: gray-world. Keeps walls neutral.
    """
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def highlight_rolloff(img_bgr: np.ndarray) -> np.ndarray:
    """
    Soft highlight compression to avoid washed/veiled whites.
    """
    x = img_bgr.astype(np.float32) / 255.0
    # Reinhard-like compression
    y = x / (x + 0.35)
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return out


def local_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE on L channel for crispness (AutoHDR-like).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def s_curve(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gentle contrast curve (prevents flat HDR look).
    """
    x = img_bgr.astype(np.float32) / 255.0
    # mild S-curve
    y = 1 / (1 + np.exp(-8 * (x - 0.5)))
    # blend curve with original to keep natural
    y = 0.55 * x + 0.45 * y
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return out


def window_mask_blend(fused_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic window pull:
    - Find very bright low-sat areas (blown windows)
    - Blend in underexposed image only in those regions
    """
    hsv = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # mask blown highlights: high V + low S
    mask = ((v > 220) & (s < 70)).astype(np.uint8) * 255

    # Expand and soften the mask
    mask = cv2.medianBlur(mask, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=6)

    # Convert to float alpha
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    alpha = np.clip(alpha * 1.25, 0, 1)  # stronger pull

    fused_f = fused_bgr.astype(np.float32)
    under_f = under_bgr.astype(np.float32)

    out = fused_f * (1 - alpha) + under_f * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_aware_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    """
    Sharpen without halos.
    """
    # Unsharp mask
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.1, sigmaY=1.1)
    sharp = cv2.addWeighted(img_bgr, 1.35, blur, -0.35, 0)

    # light detail enhance (keeps it premium)
    sharp = cv2.detailEnhance(sharp, sigma_s=10, sigma_r=0.15)
    return sharp


def autohdr_like_pipeline(under_bgr: np.ndarray, base_bgr: np.ndarray, over_bgr: np.ndarray) -> np.ndarray:
    """
    Full "AutoHDR-like" pipeline.
    """
    # Ensure same size (some phones/cameras may differ by 1px)
    H, W = base_bgr.shape[:2]
    under_bgr = cv2.resize(under_bgr, (W, H), interpolation=cv2.INTER_AREA)
    over_bgr = cv2.resize(over_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # Align under/over to base
    under_a = align_ecc(base_bgr, under_bgr)
    over_a = align_ecc(base_bgr, over_bgr)

    # Merge on scaled version (speed/memory), but keep a path to better sharpness:
    base_s, scale = resize_to_max(base_bgr, MAX_MERGE_DIM)
    under_s = cv2.resize(under_a, (base_s.shape[1], base_s.shape[0]), interpolation=cv2.INTER_AREA)
    over_s = cv2.resize(over_a, (base_s.shape[1], base_s.shape[0]), interpolation=cv2.INTER_AREA)

    fused_s = merge_mertens([under_s, base_s, over_s])

    # Post-processing on merged image
    fused_s = gray_world_wb(fused_s)
    fused_s = highlight_rolloff(fused_s)
    fused_s = local_contrast(fused_s)
    fused_s = s_curve(fused_s)

    # Window pull (blend under exposure into window areas)
    fused_s = window_mask_blend(fused_s, under_s)

    # Upscale back to full size
    fused = cv2.resize(fused_s, (W, H), interpolation=cv2.INTER_CUBIC)

    # Final crispness (apply at full res)
    fused = edge_aware_sharpen(fused)

    return fused


# =========================
# JOB LOGIC
# =========================
def fetch_next_job():
    # queued jobs with no locked_at
    # NOTE: This assumes your schema: hdr_jobs: id, set_id, status, attempts, locked_at
    rows = rest_get("hdr_jobs", "status=eq.queued&locked_at=is.null&order=created_at.asc&limit=1")
    return rows[0] if rows else None


def order_done(order_id: str):
    sets = rest_get("hdr_sets", f"order_id=eq.{order_id}&limit=200")
    # done when all are complete
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

    # Lock job
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

        # Get file rows
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

        # Main pipeline
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

        # Mark job error (or requeue if attempts remain)
        try:
            if attempts + 1 >= MAX_ATTEMPTS:
                patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
                try:
                    patch_row("hdr_sets", set_id, {"status": "error"})
                except Exception:
                    pass
            else:
                # unlock + requeue
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
