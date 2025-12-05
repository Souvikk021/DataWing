import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean
import pandas as pd

# ---------------- Config ----------------
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
ALLOWED_EXT = {'png','jpg','jpeg','tif','tiff'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'replace-with-a-random-secret'  # change this for production

# ---------------- Helpers & extractor (all 10 features) ----------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def group_components_into_lines(centroids, heights):
    if len(centroids) == 0:
        return []
    idxs = sorted(range(len(centroids)), key=lambda i: centroids[i][1])
    lines = []
    current_line = [idxs[0]]
    for i in idxs[1:]:
        prev = current_line[-1]
        y_prev = centroids[prev][1]
        y_i = centroids[i][1]
        med_h = np.median(heights) if len(heights)>0 else 20
        if abs(y_i - y_prev) < (med_h * 0.8):
            current_line.append(i)
        else:
            lines.append(current_line)
            current_line = [i]
    lines.append(current_line)
    return lines

def extract_static_features_v2(img_bgr, image_name="uploaded"):
    # Convert to grayscale
    gray = img_bgr if len(img_bgr.shape)==2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Binarize (foreground ink = 255)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh_inv.copy()

    # Connected components (foreground areas)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((thresh>0).astype('uint8'), connectivity=8)
    comps = []
    for i in range(1, num_labels):  # skip background
        x,y,w,h,a = stats[i]
        if a < 10:  # ignore tiny noise
            continue
        comps.append({
            'idx': i,
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'area': int(a),
            'cx': float(centroids[i][0]), 'cy': float(centroids[i][1])
        })

    # If no components found, return zeros + images
    if len(comps) == 0:
        features_zero = {k:0.0 for k in [
            'mean_letter_height','std_letter_height','mean_letter_width','std_letter_width',
            'letter_spacing','word_spacing','slant_angle_deg','skew_angle_deg',
            'stroke_length_total','corner_count','broken_links_count','margin_alignment_std',
            'letter_height_cv','proportion_consistency_std','horizontal_regularity','vertical_regularity',
            'num_components','total_ink_area'
        ]}
        return features_zero, {'gray': gray, 'thresh': thresh, 'skeleton': np.zeros_like(thresh), 'labels': labels}

    heights = np.array([c['h'] for c in comps])
    widths  = np.array([c['w'] for c in comps])
    lefts   = np.array([c['x'] for c in comps])
    centroids_xy = [(c['cx'], c['cy']) for c in comps]
    areas = np.array([c['area'] for c in comps])

    # 1,7,9: letter size / irregularity / proportions
    mean_h = float(np.mean(heights))
    std_h  = float(np.std(heights))
    mean_w = float(np.mean(widths))
    std_w  = float(np.std(widths))
    letter_height_cv = float(std_h / (mean_h + 1e-9))
    proportion_ratios = widths / (heights + 1e-9)
    proportion_consistency_std = float(np.std(proportion_ratios))

    # 2: spacing (left->right)
    comps_sorted_lr = sorted(comps, key=lambda c: c['x'])
    gaps = []
    for i in range(len(comps_sorted_lr)-1):
        x1 = comps_sorted_lr[i]['x']; w1 = comps_sorted_lr[i]['w']
        x2 = comps_sorted_lr[i+1]['x']
        gap = x2 - (x1 + w1)
        gaps.append(gap)
    gaps = np.array(gaps) if len(gaps)>0 else np.array([])
    letter_spacing = float(np.percentile(gaps, 30)) if gaps.size>0 else 0.0
    word_spacing   = float(np.percentile(gaps, 80)) if gaps.size>0 else 0.0

    # 3: slant via Hough lines on edges
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=10)
    slant_angle_deg = 0.0
    if lines is not None and len(lines)>0:
        angles = []
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            angles.append(np.arctan2((y2-y1),(x2-x1)))
        slant_angle_deg = float(np.degrees(np.mean(angles)))

    # 4: skewness via minAreaRect
    coords = np.column_stack(np.where(thresh > 0))
    skew_angle_deg = 0.0
    if coords.shape[0] >= 10:
        rect = cv2.minAreaRect(coords.astype(np.int32))
        skew_angle_deg = float(rect[-1])

    # 6: stroke length via skeleton
    bw = (thresh > 0)
    skel = skeletonize(bw)
    skel_u8 = img_as_ubyte(skel)
    ys, xs = np.where(skel_u8 > 0)
    stroke_length_total = 0.0
    if len(xs)>1:
        order = np.lexsort((xs, ys))
        xs2 = xs[order]; ys2 = ys[order]
        for i in range(len(xs2)-1):
            stroke_length_total += euclidean((xs2[i], ys2[i]), (xs2[i+1], ys2[i+1]))

    # 8: corners and broken links
    img_u8 = (thresh>0).astype('uint8')*255
    corners = cv2.goodFeaturesToTrack(img_u8, maxCorners=1000, qualityLevel=0.01, minDistance=6)
    corner_count = 0 if corners is None else int(len(corners))

    # group components into lines
    line_groups = group_components_into_lines(centroids=[(c['cx'],c['cy']) for c in comps], heights=heights.tolist())
    baselines = []
    verticals = []
    left_margin_per_line = []
    broken_links_count = 0
    for lg in line_groups:
        ys_line = [comps[i]['cy'] for i in lg]
        baselines.append(np.mean(ys_line))
        heights_line = [comps[i]['h'] for i in lg]
        verticals.append(np.std(heights_line) if len(heights_line)>0 else 0.0)
        lefts_line = [comps[i]['x'] for i in lg]
        left_margin_per_line.append(np.min(lefts_line) if len(lefts_line)>0 else 0.0)
        comps_line = sorted([comps[i] for i in lg], key=lambda c:c['x'])
        gaps_line = []
        for j in range(len(comps_line)-1):
            gap = comps_line[j+1]['x'] - (comps_line[j]['x'] + comps_line[j]['w'])
            gaps_line.append(gap)
        if len(gaps_line)>0:
            med_h_line = np.median([c['h'] for c in comps_line]) if len(comps_line)>0 else 0
            for g in gaps_line:
                if g > max(10, 0.6*med_h_line):
                    broken_links_count += 1

    horizontal_regularity = float(np.std(baselines)) if len(baselines)>0 else 0.0
    vertical_regularity = float(np.mean(verticals)) if len(verticals)>0 else 0.0
    margin_alignment_std = float(np.std(left_margin_per_line)) if len(left_margin_per_line)>0 else 0.0

    num_components = int(len(comps))
    total_ink_area = float(np.sum(areas))

    features = {
        'image_name': image_name,
        'mean_letter_height': mean_h,
        'std_letter_height': std_h,
        'mean_letter_width': mean_w,
        'std_letter_width': std_w,
        'letter_spacing': letter_spacing,
        'word_spacing': word_spacing,
        'slant_angle_deg': slant_angle_deg,
        'skew_angle_deg': skew_angle_deg,
        'horizontal_regularity_baseline_std': horizontal_regularity,
        'vertical_regularity_height_std_mean': vertical_regularity,
        'stroke_length_total': float(stroke_length_total),
        'letter_height_cv': letter_height_cv,
        'corner_count': corner_count,
        'broken_links_count': int(broken_links_count),
        'proportion_consistency_std': proportion_consistency_std,
        'margin_alignment_std': margin_alignment_std,
        'num_components': num_components,
        'total_ink_area': total_ink_area
    }

    images_out = {'gray': gray, 'thresh': thresh, 'skeleton': skel_u8, 'labels': labels}
    return features, images_out

# ---------------- Routes ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.',1)[1].lower()
        uid = str(uuid.uuid4())[:8]
        fname = f"{uid}.{ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(upload_path)
        # read image robustly
        img = cv2.imdecode(np.fromfile(upload_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(upload_path)
        if img is None:
            flash('Unable to read uploaded image')
            return redirect(url_for('index'))
        feats, images = extract_static_features_v2(img)
        base = uid
        gray_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base}_gray.png")
        thresh_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base}_thresh.png")
        skel_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base}_skel.png")
        cv2.imwrite(gray_path, images['gray'])
        cv2.imwrite(thresh_path, images['thresh'])
        cv2.imwrite(skel_path, images['skeleton'])
        return render_template('result.html',
                               upload_url = url_for('static', filename=f"uploads/{fname}"),
                               gray_url = url_for('static', filename=f"processed/{base}_gray.png"),
                               thresh_url = url_for('static', filename=f"processed/{base}_thresh.png"),
                               skel_url = url_for('static', filename=f"processed/{base}_skel.png"),
                               features = feats)
    else:
        flash('Invalid file type')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
