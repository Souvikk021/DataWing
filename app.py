import os
import uuid
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF

from flask import Flask, render_template, request, url_for
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
ALLOWED_EXT = {"png", "jpg", "jpeg", "tif", "tiff", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)

# ---------------- HELPERS ----------------
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def imwrite_safe(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


def pixmap_to_cv2(pix):
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        arr = arr.reshape(pix.h, pix.w, 4)[:, :, :3]
    else:
        arr = arr.reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------- DYSGRAPHIA SCREENING ----------------
def dysgraphia_screening(f):
    score = 0

    if f["letter_height_cv"] > 0.65:
        score += 1
    if f["horizontal_regularity_baseline_std"] > 1.8:
        score += 1
    if f["letter_spacing"] < -45:
        score += 1
    if f["corner_density"] > 0.02:
        score += 1

    label = (
        "Dysgraphic-like indicators present"
        if score >= 3
        else "No strong dysgraphic indicators"
    )
    return label, score


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    comps = []
    for i in range(1, num_labels):
        x, y, w, h, a = stats[i]
        if a > 20:
            comps.append((x, y, w, h, a, centroids[i]))

    heights = np.array([c[3] for c in comps])
    widths = np.array([c[2] for c in comps])
    areas = np.array([c[4] for c in comps])
    lefts = np.array([c[0] for c in comps])

    mean_h = float(np.mean(heights)) if len(heights) else 0
    std_h = float(np.std(heights)) if len(heights) else 0
    mean_w = float(np.mean(widths)) if len(widths) else 0
    std_w = float(np.std(widths)) if len(widths) else 0

    letter_height_cv = std_h / (mean_h + 1e-6)

    proportions = widths / (heights + 1e-6)
    proportion_consistency_std = float(np.std(proportions)) if len(proportions) else 0

    comps_lr = sorted(comps, key=lambda c: c[0])
    gaps = [
        comps_lr[i + 1][0] - (comps_lr[i][0] + comps_lr[i][2])
        for i in range(len(comps_lr) - 1)
    ]

    letter_spacing = float(np.percentile(gaps, 30)) if gaps else 0
    word_spacing = float(np.percentile(gaps, 80)) if gaps else 0

    skel = skeletonize(thresh > 0)
    skel_u8 = img_as_ubyte(skel)
    ys, xs = np.where(skel_u8 > 0)
    stroke_len = sum(
        euclidean((xs[i], ys[i]), (xs[i + 1], ys[i + 1]))
        for i in range(len(xs) - 1)
    ) if len(xs) > 1 else 0

    corners = cv2.goodFeaturesToTrack(
        thresh, maxCorners=500, qualityLevel=0.01, minDistance=6
    )
    corner_count = 0 if corners is None else len(corners)

    total_ink_area = float(np.sum(thresh > 0))
    corner_density = corner_count / (total_ink_area + 1e-6)

    baseline_std = np.std([c[5][1] for c in comps]) / (mean_h + 1e-6)
    vertical_regularity_height_std = float(np.std(heights)) if len(heights) else 0
    margin_alignment_std = float(np.std(lefts)) if len(lefts) else 0

    num_components = len(comps)
    component_density = num_components / (total_ink_area + 1e-6)

    features = {
        "image_name": name,

        # Size
        "mean_letter_height": mean_h,
        "std_letter_height": std_h,
        "mean_letter_width": mean_w,
        "std_letter_width": std_w,
        "letter_height_cv": letter_height_cv,
        "proportion_consistency_std": proportion_consistency_std,

        # Spacing
        "letter_spacing": letter_spacing,
        "word_spacing": word_spacing,

        # Stroke & shape
        "stroke_length_total": float(stroke_len),
        "corner_count": corner_count,
        "corner_density": corner_density,

        # Structure
        "num_components": num_components,
        "total_ink_area": total_ink_area,
        "component_density": component_density,

        # Alignment & regularity
        "horizontal_regularity_baseline_std": float(baseline_std),
        "vertical_regularity_height_std": vertical_regularity_height_std,
        "margin_alignment_std": margin_alignment_std,

        # Slant
        "slant_angle_deg": 0.0
    }

    label, score = dysgraphia_screening(features)
    features["dysgraphia_label"] = label
    features["dysgraphia_score"] = score

    return features, gray, thresh, skel_u8


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    results = []
    all_features = []

    for f in files:
        if not allowed_file(f.filename):
            continue

        name, ext = os.path.splitext(f.filename)

        # -------- PDF --------
        if ext.lower() == ".pdf":
            doc = fitz.open(stream=f.read(), filetype="pdf")
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                img = pixmap_to_cv2(pix)

                sample = f"{name}_page{i+1}"
                feats, g, t, s = extract_features(img, sample)
                feats["source_file"] = f.filename

                imwrite_safe(os.path.join(UPLOAD_FOLDER, f"{sample}.png"), img)
                imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{sample}_g.png"), g)
                imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{sample}_t.png"), t)
                imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{sample}_s.png"), s)

                all_features.append(feats)
                results.append({
                    "sample_name": sample,
                    "source_filename": f.filename,
                    "upload_url": url_for("static", filename=f"uploads/{sample}.png"),
                    "gray_url": url_for("static", filename=f"processed/{sample}_g.png"),
                    "thresh_url": url_for("static", filename=f"processed/{sample}_t.png"),
                    "skel_url": url_for("static", filename=f"processed/{sample}_s.png"),
                    "features": feats
                })

        # -------- SINGLE IMAGE --------
        else:
            uid = uuid.uuid4().hex[:8]
            path = os.path.join(UPLOAD_FOLDER, uid + ext)
            f.save(path)

            img = cv2.imread(path)
            feats, g, t, s = extract_features(img, uid)
            feats["source_file"] = f.filename

            imwrite_safe(os.path.join(UPLOAD_FOLDER, f"{uid}.png"), img)
            imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{uid}_g.png"), g)
            imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{uid}_t.png"), t)
            imwrite_safe(os.path.join(PROCESSED_FOLDER, f"{uid}_s.png"), s)

            all_features.append(feats)
            results.append({
                "sample_name": uid,
                "source_filename": f.filename,
                "upload_url": url_for("static", filename=f"uploads/{uid}.png"),
                "gray_url": url_for("static", filename=f"processed/{uid}_g.png"),
                "thresh_url": url_for("static", filename=f"processed/{uid}_t.png"),
                "skel_url": url_for("static", filename=f"processed/{uid}_s.png"),
                "features": feats
            })

    csv_path = os.path.join(PROCESSED_FOLDER, "features.csv")
    pd.DataFrame(all_features).to_csv(csv_path, index=False)

    return render_template(
        "result.html",
        results=results,
        csv_url=url_for("static", filename="processed/features.csv")
    )


if __name__ == "__main__":
    app.run(debug=True, port=8501)
