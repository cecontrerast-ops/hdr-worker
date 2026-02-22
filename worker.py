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
    # if someone accidentally stored "hdr_raw/uuid/file.jpg", strip "hdr_raw/"
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

    # 🔥 IMMEDIATE DOWNSCALE right after decode (saves RAM)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1500:
        scale = 1500 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

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


# =========================
# HDR FUSION (FIXED: NO BLACK OUTPUT)
# =========================
def exposure_fusion(imgs_bgr):
    try:
        if len(imgs_bgr) < 3:
            return imgs_bgr[1]

        # Ensure same size
        h, w = imgs_bgr[0].shape[:2]
        imgs = [cv2.resize(img, (w, h)) for img in imgs_bgr]

        # Convert to float 0..1
        imgs_float = [(img.astype(np.float32) / 255.0) for img in imgs]

        mertens = cv2.createMergeMertens()
        fused = mertens.process(imgs_float)  # float32

        # ✅ Clamp
        fused = np.clip(fused, 0.0, 1.0)

        # ✅ Normalize brightness so it doesn't come out near-black
        mx = float(fused.max())
        if mx > 1e-6:
            fused = fused / mx

        # ✅ Gamma correction (screen-friendly)
        gamma = 1.0 / 2.2
        fused = np.power(fused, gamma)

        out = (fused * 255.0).clip(0, 255).astype(np.uint8)

        # free memory
        del imgs_float
        del fused
        del imgs

        return out

    except Exception as e:
        print("FUSION ERROR:", str(e))
        return imgs_bgr[1]


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

        # DOWNLOAD
        img_under = storage_download(RAW_BUCKET, f_under["storage_path"])
        img_base = storage_download(RAW_BUCKET, f_base["storage_path"])
        img_over = storage_download(RAW_BUCKET, f_over["storage_path"])

        # FUSION
        out = exposure_fusion([img_under, img_base, img_over])

        out_path = f"{order_id}/{set_id}.jpg"

        # UPLOAD
        storage_upload(OUT_BUCKET, out_path, out)

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
