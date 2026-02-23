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

RAW_BUCKET = os.getenv("RAW_BUCKET", "hdr_raw")
OUT_BUCKET = os.getenv("OUT_BUCKET", "hdr_output")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "6"))

# HDR tuning (safe + closer to real estate)
MAX_DIM = int(os.getenv("MAX_DIM", "2200"))  # downscale longest side to this
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))

# If your RAW bucket is PUBLIC, this is best.
# If your RAW bucket is PRIVATE, this still works because we try authenticated endpoints too.
PREFER_PUBLIC_READ = os.getenv("PREFER_PUBLIC_READ", "true").lower() in ("1", "true", "yes")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

HEADERS_JSON = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
    # IMPORTANT: makes PostgREST return JSON on PATCH/POST when possible
    "Prefer": "return=representation",
}

HEADERS_AUTH_ONLY = {
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
}

print("=== HDR WORKER v4 (STORAGE FIX + SAFE HDR) LOADED ===")


# =========================
# HELPERS
# =========================
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_json(resp: requests.Response):
    """
    PostgREST sometimes returns empty body on PATCH depending on headers.
    This avoids JSONDecodeError crashes.
    """
    if resp.content is None or len(resp.content) == 0:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def sb_get(path: str):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=HEADERS_JSON, timeout=30)
    r.raise_for_status()
    return r.json()


def sb_patch(path: str, payload: dict):
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=HEADERS_JSON, json=payload, timeout=30)
    # If something fails, show the body for debugging
    if r.status_code >= 400:
        raise RuntimeError(f"PATCH failed {r.status_code}: {r.text}")
    return _safe_json(r)


def patch_row(table: str, row_id: str, payload: dict):
    # PostgREST update by id
    return sb_patch(f"/rest/v1/{table}?id=eq.{row_id}", payload)


def get_next_job():
    """
    Find a queued job, lock it.
    """
    # Get queued jobs with lowest attempts first
    jobs = sb_get(
        f"/rest/v1/hdr_jobs?"
        f"select=id,set_id,status,attempts,locked_at&"
        f"status=eq.queued&"
        f"order=attempts.asc,created_at.asc&"
        f"limit=1"
    )
    if not jobs:
        return None

    job = jobs[0]
    if (job.get("attempts") or 0) >= MAX_ATTEMPTS:
        # Mark as error to prevent infinite loop
        patch_row("hdr_jobs", job["id"], {"status": "error", "last_error": "MAX_ATTEMPTS reached"})
        return None

    # Lock it
    patch_row(
        "hdr_jobs",
        job["id"],
        {"locked_at": now_iso(), "status": "processing", "attempts": (job.get("attempts") or 0) + 1},
    )
    return job


def get_set(set_id: str):
    rows = sb_get(f"/rest/v1/hdr_sets?select=*&id=eq.{set_id}&limit=1")
    if not rows:
        raise RuntimeError(f"hdr_sets not found: {set_id}")
    return rows[0]


def get_file(file_id: str):
    rows = sb_get(f"/rest/v1/hdr_files?select=*&id=eq.{file_id}&limit=1")
    if not rows:
        raise RuntimeError(f"hdr_files not found: {file_id}")
    return rows[0]


def order_done(order_id: str):
    sets = sb_get(f"/rest/v1/hdr_sets?select=id,status&order_id=eq.{order_id}")
    if not sets:
        return True
    for s in sets:
        if s.get("status") != "complete":
            return False
    return True


# =========================
# STORAGE (THIS WAS THE BUG)
# =========================
def _download_bytes_from_storage(bucket: str, storage_path: str) -> bytes:
    """
    Try multiple Supabase Storage URL shapes to avoid 400/black output issues.
    """
    # Normalize path (no leading slash)
    p = storage_path.lstrip("/")

    candidates = []

    # Public read (fastest when bucket is public)
    if PREFER_PUBLIC_READ:
        candidates.append(f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{p}")

    # Authenticated read path
    candidates.append(f"{SUPABASE_URL}/storage/v1/object/authenticated/{bucket}/{p}")

    # Direct object path (works when using service role + policies allow)
    candidates.append(f"{SUPABASE_URL}/storage/v1/object/{bucket}/{p}")

    last_err = None
    for url in candidates:
        r = requests.get(url, headers=HEADERS_AUTH_ONLY, timeout=60)
        if r.status_code == 200 and r.content:
            return r.content
        last_err = f"{url} -> {r.status_code} {r.text[:200]}"

    raise RuntimeError(f"Download failed for {bucket}/{p}. Last error: {last_err}")


def download_image(storage_path: str) -> np.ndarray:
    data = _download_bytes_from_storage(RAW_BUCKET, storage_path)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("OpenCV failed to decode downloaded image (img is None)")
    return img


def upload_image(storage_path: str, img_bgr: np.ndarray):
    """
    Upload JPEG to OUT_BUCKET using service role.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise RuntimeError("upload_image got empty image")

    # Ensure uint8
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    p = storage_path.lstrip("/")

    url = f"{SUPABASE_URL}/storage/v1/object/{OUT_BUCKET}/{p}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "image/jpeg",
        "x-upsert": "true",
    }

    r = requests.post(url, headers=headers, data=buf.tobytes(), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text}")


# =========================
# IMAGE PIPELINE
# =========================
def resize_longest(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def simple_grayworld_wb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mild white balance to reduce orange/blue casts.
    """
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    m = (mb + mg + mr) / 3.0
    b *= (m / (mb + 1e-6))
    g *= (m / (mg + 1e-6))
    r *= (m / (mr + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def clarity_boost(img_bgr: np.ndarray, amount: float = 0.22) -> np.ndarray:
    """
    Subtle micro-contrast (helps "soft" look).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = l.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(l, (0, 0), 2.0)
    l2 = l + amount * (l - blur)
    l2 = np.clip(l2, 0, 1)

    l_out = (l2 * 255.0).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)
    return out


def window_mask_blend(base_bgr: np.ndarray, under_bgr: np.ndarray) -> np.ndarray:
    """
    Window pull blend:
    - Detect bright window areas in base and replace with under exposure.
    """
    base = base_bgr.astype(np.float32)
    under = under_bgr.astype(np.float32)

    # Luma from base
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold for bright areas (tune)
    # (v > 210) gives stronger pulls than 220
    mask = (gray > 210).astype(np.float32)

    # Feather mask
    mask = cv2.GaussianBlur(mask, (0, 0), 6.0)

    # Stronger alpha
    alpha = np.clip(mask * 1.5, 0, 1)  # stronger than 1.25

    alpha3 = cv2.merge([alpha, alpha, alpha])

    out = base * (1 - alpha3) + under * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def exposure_fusion(under_bgr: np.ndarray, mid_bgr: np.ndarray, over_bgr: np.ndarray) -> np.ndarray:
    """
    Safe Mertens exposure fusion + real estate finishing:
    - downscale for memory
    - fusion
    - mild WB
    - window pulls
    - micro-contrast
    """
    # Resize for safety + consistency
    under = resize_longest(under_bgr, MAX_DIM)
    mid = resize_longest(mid_bgr, MAX_DIM)
    over = resize_longest(over_bgr, MAX_DIM)

    # Ensure same size
    h, w = mid.shape[:2]
    under = cv2.resize(under, (w, h), interpolation=cv2.INTER_AREA)
    over = cv2.resize(over, (w, h), interpolation=cv2.INTER_AREA)

    # Mertens wants float 0..1
    imgs_f = [x.astype(np.float32) / 255.0 for x in [under, mid, over]]
    mertens = cv2.createMergeMertens()
    fused = mertens.process(imgs_f)  # float32 0..1, 3ch

    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    # Mild WB to reduce mixed lighting look
    out = simple_grayworld_wb(out)

    # Window pulls (use under)
    out = window_mask_blend(out, under)

    # Clarity / sharpness
    out = clarity_boost(out, amount=0.24)

    return out


# =========================
# JOB LOOP
# =========================
def process_once() -> bool:
    job = get_next_job()
    if not job:
        print("Worker alive: checking for jobs...")
        return False

    job_id = job["id"]
    set_id = job["set_id"]
    print(f"Picked job {job_id} set_id={set_id} attempts={job.get('attempts', 0)}")

    try:
        s = get_set(set_id)
        order_id = s["order_id"]

        patch_row("hdr_sets", set_id, {"status": "processing"})
        patch_row("hdr_orders", order_id, {"status": "processing"})

        f_under = get_file(s["file_under_id"])
        f_base = get_file(s["file_base_id"])
        f_over = get_file(s["file_over_id"])

        print("Paths:", f_under["storage_path"], f_base["storage_path"], f_over["storage_path"])

        print("Downloading:", f"{RAW_BUCKET}/{f_under['storage_path']}")
        img_under = download_image(f_under["storage_path"])

        print("Downloading:", f"{RAW_BUCKET}/{f_base['storage_path']}")
        img_base = download_image(f_base["storage_path"])

        print("Downloading:", f"{RAW_BUCKET}/{f_over['storage_path']}")
        img_over = download_image(f_over["storage_path"])

        out = exposure_fusion(img_under, img_base, img_over)

        out_path = f"{order_id}/{set_id}.jpg"
        print("Uploading:", f"{OUT_BUCKET}/{out_path}")
        upload_image(out_path, out)

        patch_row("hdr_sets", set_id, {"status": "complete", "output_path": out_path})
        patch_row("hdr_jobs", job_id, {"status": "complete"})

        if order_done(order_id):
            patch_row("hdr_orders", order_id, {"status": "complete"})

        print(f"Completed set {set_id}")
        return True

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        print("ERROR:", err)
        traceback.print_exc()

        try:
            patch_row("hdr_jobs", job_id, {"status": "error", "last_error": err})
        except Exception:
            pass

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
