import os
import time
import json
import math
import requests
import numpy as np
import cv2
from datetime import datetime, timezone

print("=== NEW WORKER VERSION LOADED (HDR) ===")

# ----------------------------
# ENV
# ----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.environ.get("OUT_BUCKET", "hdr_output")

POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "2"))
LOCK_SECONDS = int(os.environ.get("LOCK_SECONDS", "180"))
MAX_LONG_EDGE = int(os.environ.get("MAX_LONG_EDGE", "2400"))  # 2000–2600 recommended for Railway RAM
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

# HDR tuning knobs (you can adjust without redesign)
HIGHLIGHT_ROLLOFF_DEN = float(os.environ.get("HIGHLIGHT_ROLLOFF_DEN", "0.30"))  # 0.35 -> darker, 0.30 -> brighter
SCURVE_BLEND_A = float(os.environ.get("SCURVE_BLEND_A", "0.45"))  # 0.55/0.45 -> darker, 0.45/0.55 -> brighter
SCURVE_BLEND_B = 1.0 - SCURVE_BLEND_A

WINDOW_V_THRESH = int(os.environ.get("WINDOW_V_THRESH", "210"))  # 220 -> weaker pull, 210 -> stronger pull
WINDOW_ALPHA_MULT = float(os.environ.get("WINDOW_ALPHA_MULT", "1.50"))  # 1.25 -> weaker, 1.5 -> stronger

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("FATAL: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
    raise SystemExit(1)

# ----------------------------
# Helpers: safe HTTP / Supabase REST
# ----------------------------
def _safe_json(resp: requests.Response):
    """Return JSON dict/list if present; return {} on empty body."""
    txt = (resp.text or "").strip()
    if not txt:
        return {}
    try:
        return resp.json()
    except Exception:
        # Sometimes Supabase returns empty or non-JSON
        return {}

def sb_get(path: str):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return _safe_json(r)

def sb_post(path: str, body):
    url = f"{SUPABASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return _safe_json(r)

def sb_patch(path: str, body):
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    # PATCH often returns 204 No Content
    if r.status_code >= 400:
        # include body for debugging
        raise RuntimeError(f"Supabase PATCH failed {r.status_code}: {r.text[:500]}")
    return _safe_json(r)

def patch_row(table: str, row_id: str, fields: dict):
    path = f"/rest/v1/{table}?id=eq.{row_id}"
    # Prefer: return=representation (sometimes), but safe even if 204
    headers = dict(HEADERS)
    headers["Prefer"] = "return=representation"
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=headers, data=json.dumps(fields), timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"PATCH {table} {row_id} failed {r.status_code}: {r.text[:500]}")
    return _safe_json(r)

def fetch_one(table: str, row_id: str):
    data = sb_get(f"/rest/v1/{table}?id=eq.{row_id}&select=*&limit=1")
    if isinstance(data, list) and data:
        return data[0]
    return None

# ----------------------------
# Storage: signed URLs for service role
# ----------------------------
def storage_download(bucket: str, path: str) -> np.ndarray:
    # Download object
    # Using /storage/v1/object/{bucket}/{path}
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.get(url, headers={"Authorization": HEADERS["Authorization"]}, timeout=60)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {bucket}/{path}")
    return img

def storage_upload_jpg(bucket: str, path: str, bgr_img: np.ndarray):
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.post(
        url,
        headers={
            "Authorization": HEADERS["Authorization"],
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        },
        data=buf.tobytes(),
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text[:500]}")
    return True

# ----------------------------
# Image utilities (quality)
# ----------------------------
def resize_long_edge(img: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def estimate_roll_angle_deg(img_bgr: np.ndarray) -> float:
    """Estimate small roll rotation to make verticals straighter (not full keystone)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
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
        angle = math.degrees(math.atan2(dy, dx))  # -180..180 (0=horizontal)
        # Keep near-vertical lines only
        # vertical = 90 or -90
        a = abs(abs(angle) - 90.0)
        if a < 15.0:
            # convert to roll deviation: if line is 92, image needs -2 degrees
            # sign: angle>90 => rotate - (angle-90)
            dev = (angle - 90.0) if angle > 0 else (angle + 90.0)
            # clamp
            if abs(dev) <= 6.0:
                angles.append(dev)

    if len(angles) < 6:
        return 0.0

    return float(np.median(angles))

def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.15:
        return img
    (h, w) = img.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def highlight_rolloff(img_bgr: np.ndarray, den: float = 0.30) -> np.ndarray:
    """Compress highlights gently to avoid washed out whites."""
    x = img_bgr.astype(np.float32) / 255.0
    # rolloff on luminance-ish (operate per channel but mild)
    # y = x / (x + den) normalized
    y = x / (x + den)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

def s_curve(img_bgr: np.ndarray) -> np.ndarray:
    """Mild contrast curve without crushing shadows."""
    x = img_bgr.astype(np.float32) / 255.0
    # classic smoothstep-ish curve
    y = x * x * (3.0 - 2.0 * x)
    y = np.clip(y, 0.0, 1.0)
    out = (y * 255.0).astype(np.uint8)
    # blend with original
    return cv2.addWeighted(img_bgr, SCURVE_BLEND_A, out, SCURVE_BLEND_B, 0)

def clarity_boost(img_bgr: np.ndarray, amount: float = 0.30) -> np.ndarray:
    """Local contrast boost (clarity)"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    L_out = cv2.addWeighted(L, 1.0 - amount, L2, amount, 0)
    lab2 = cv2.merge([L_out, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def sharpen(img_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Unsharp mask"""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    return cv2.addWeighted(img_bgr, 1.0 + strength, blur, -strength, 0)

def window_mask_blend(base_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    """
    Stronger window pulls:
    - detect bright window regions in base
    - blend in underexposed detail
    """
    hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # mask: very bright areas likely windows
    mask = (v > float(WINDOW_V_THRESH)).astype(np.float32)

    # soften mask edges
    mask = cv2.GaussianBlur(mask, (0, 0), 7.0)

    # strengthen alpha, cap to avoid halos
    alpha = np.clip(mask * WINDOW_ALPHA_MULT, 0.0, 0.85)

    # blend
    base_f = base_bgr.astype(np.float32)
    under_f = under_bgr.astype(np.float32)
    out = (base_f * (1.0 - alpha[..., None]) + under_f * alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

def exposure_fusion_safe(imgs_bgr) -> np.ndarray:
    """
    Safe Mertens fusion:
    - downscale to reduce RAM
    - ensure same size
    """
    if not imgs_bgr or len(imgs_bgr) < 3:
        raise RuntimeError("Need 3 bracket images")

    # resize each to max long edge
    resized = [resize_long_edge(im, MAX_LONG_EDGE) for im in imgs_bgr]

    # ensure same size
    h, w = resized[0].shape[:2]
    resized = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in resized]

    # MergeMertens expects float32 0..1
    mertens = cv2.createMergeMertens()
    imgs_f = [im.astype(np.float32) / 255.0 for im in resized]
    fused = mertens.process(imgs_f)  # float32 0..1
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return out

def post_process(base_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline to get closer to AutoHDR look:
    1) highlight rolloff (avoid washout)
    2) window mask blend (pull windows)
    3) s-curve contrast
    4) clarity + sharpen
    """
    x = highlight_rolloff(base_bgr, den=HIGHLIGHT_ROLLOFF_DEN)
    x = window_mask_blend(x, under_bgr)
    x = s_curve(x)
    x = clarity_boost(x, amount=0.30)
    x = sharpen(x, strength=0.55)
    return x

# ----------------------------
# Job Logic
# ----------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def pick_job():
    # Find unlocked queued job
    # status=queued AND (locked_at is null OR locked_at < now-LOCK_SECONDS)
    # NOTE: locked_at stored as timestamptz in DB recommended
    # We'll use simple query and filter client-side to avoid tricky RPC.
    jobs = sb_get("/rest/v1/hdr_jobs?select=*&status=eq.queued&order=created_at.asc&limit=10")
    if not isinstance(jobs, list) or not jobs:
        return None

    # pick first not locked recently
    now_ts = datetime.now(timezone.utc).timestamp()
    for j in jobs:
        locked_at = j.get("locked_at")
        if not locked_at:
            return j
        try:
            # parse ISO (best effort)
            t = datetime.fromisoformat(locked_at.replace("Z", "+00:00")).timestamp()
            if (now_ts - t) > LOCK_SECONDS:
                return j
        except Exception:
            # if parsing fails, treat as unlocked
            return j
    return None

def order_done(order_id: str) -> bool:
    sets = sb_get(f"/rest/v1/hdr_sets?select=id,status&order_id=eq.{order_id}&limit=2000")
    if not isinstance(sets, list) or not sets:
        return False
    for s in sets:
        if s.get("status") not in ("complete", "error"):
            return False
    return True

def process_once() -> bool:
    job = pick_job()
    if not job:
        return False

    job_id = job["id"]
    set_id = job.get("set_id")
    attempts = int(job.get("attempts") or 0)

    print(f"Picked job {job_id} set_id={set_id} attempts={attempts}")

    # lock job immediately
    lock_time = now_iso()
    try:
        patch_row("hdr_jobs", job_id, {"status": "processing", "locked_at": lock_time, "attempts": attempts + 1})
    except Exception as e:
        print(f"Lock failed: {e}")
        return True  # continue loop

    try:
        # fetch set / order
        s = fetch_one("hdr_sets", set_id)
        if not s:
            raise RuntimeError(f"Set not found: {set_id}")

        order_id = s.get("order_id")
        patch_row("hdr_sets", set_id, {"status": "processing"})
        if order_id:
            patch_row("hdr_orders", order_id, {"status": "processing"})

        # load file rows
        file_under_id = s.get("file_under_id")
        file_base_id = s.get("file_base_id")
        file_over_id = s.get("file_over_id")

        f_under = fetch_one("hdr_files", file_under_id)
        f_base = fetch_one("hdr_files", file_base_id)
        f_over = fetch_one("hdr_files", file_over_id)

        if not f_under or not f_base or not f_over:
            raise RuntimeError("Missing file rows for set")

        p_under = f_under["storage_path"]
        p_base = f_base["storage_path"]
        p_over = f_over["storage_path"]

        print(f"Paths: {p_under} {p_base} {p_over}")

        # download images
        print(f"Downloading: {RAW_BUCKET}/{p_under}")
        img_under = storage_download(RAW_BUCKET, p_under)
        print(f"Downloading: {RAW_BUCKET}/{p_base}")
        img_base = storage_download(RAW_BUCKET, p_base)
        print(f"Downloading: {RAW_BUCKET}/{p_over}")
        img_over = storage_download(RAW_BUCKET, p_over)

        # optional roll correction (use base exposure as reference)
        roll = estimate_roll_angle_deg(img_base)
        if abs(roll) > 0.15:
            img_under = rotate_bound(img_under, -roll)
            img_base = rotate_bound(img_base, -roll)
            img_over = rotate_bound(img_over, -roll)

        # fuse
        fused = exposure_fusion_safe([img_under, img_base, img_over])

        # post process closer to AutoHDR (use under for windows)
        out = post_process(fused, resize_long_edge(img_under, MAX_LONG_EDGE))

        # final resize back to fused dims already; encode + upload
        out_path = f"{order_id}/{set_id}.jpg" if order_id else f"{set_id}.jpg"
        print(f"Uploading: {OUT_BUCKET}/{out_path}")
        storage_upload_jpg(OUT_BUCKET, out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete", "last_error": None})

        if order_id and order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})

        print(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = str(e)
        print(f"ERROR job {job_id}: {err}")
        try:
            patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
        except Exception as e2:
            print(f"Failed updating job error: {e2}")
        try:
            if set_id:
                patch_row("hdr_sets", set_id, {"status": "error"})
        except Exception as e3:
            print(f"Failed updating set error: {e3}")
        return True

def main():
    print("HDR Worker started.")
    while True:
        try:
            did = process_once()
            if not did:
                print("Worker alive: checking for jobs...")
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
