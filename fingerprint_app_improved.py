"""
Fingerprint Recognition (Streamlit app)
- Upload two fingerprint images (reference and probe)
- Preprocess images (CLAHE, Gabor optional)
- Skeletonize and extract minutiae (ridge endings & bifurcations)
- Visualize minutiae on the images
- Match minutiae points and compute a simple matching score
- Show matched pairs (lines) and allow downloading the annotated result

This file is written with clarity in mind (suitable for assignment submission).
"""

import io
import math
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

# ----------------------------
# Helper types
# ----------------------------
Point = Tuple[int, int]  # (x, y)

# ----------------------------
# Preprocessing functions
# ----------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or RGB image to gray."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def apply_clahe(gray: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to improve contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def gabor_enhance(gray: np.ndarray, ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5) -> np.ndarray:
    """A single-orientation Gabor filter enhancement (helpful sometimes)."""
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    # normalize result
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return filtered

def binarize_image(gray: np.ndarray) -> np.ndarray:
    """Binarize using Otsu thresholding and return binary image with 0/1 values."""
    # small blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    th = threshold_otsu(blurred)
    binary = (blurred > th).astype(np.uint8)
    # sometimes ridge is white; for skeletonize we want ridges as 1
    # if ridges are dark, invert
    # Heuristic: check mean of foreground
    if np.mean(blurred[binary == 1]) < np.mean(blurred[binary == 0]):
        binary = 1 - binary
    return binary

def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    """Skeletonize binary image (expects 0/1 array) and return 0/1 skeleton."""
    skel = skeletonize(binary).astype(np.uint8)
    return skel

# ----------------------------
# Minutiae extraction
# ----------------------------
def get_minutiae_from_skeleton(skel: np.ndarray) -> Tuple[List[Point], List[Point]]:
    """
    Detect minutiae with a simple crossing-number-like approach:
    - ridge ending: a skeleton pixel with exactly 1 neighbor
    - bifurcation: a skeleton pixel with exactly 3 neighbors
    Returns (endings, bifurcations) as lists of (x, y)
    """
    endings = []
    bifurcations = []
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 1:
                nb = np.sum(skel[y - 1 : y + 2, x - 1 : x + 2]) - 1
                if nb == 1:
                    endings.append((x, y))
                elif nb == 3:
                    bifurcations.append((x, y))
    return endings, bifurcations

# ----------------------------
# Matching functions
# ----------------------------
def match_minutiae(
    ptsA: List[Point],
    ptsB: List[Point],
    tolerance_px: int = 15
) -> Tuple[List[Tuple[Point, Point]], float]:
    """
    Very simple nearest-neighbor matching:
    For each point in A, find the nearest in B within tolerance_px.
    Return matched pairs and a score (percentage of matched points relative to average count).
    """
    if len(ptsA) == 0 and len(ptsB) == 0:
        return [], 100.0
    if len(ptsA) == 0 or len(ptsB) == 0:
        return [], 0.0

    ptsB_used = set()
    matched_pairs = []

    for a in ptsA:
        ax, ay = a
        best_dist = tolerance_px + 1
        best_b = None
        for idx, b in enumerate(ptsB):
            if idx in ptsB_used:
                continue
            bx, by = b
            d = math.hypot(ax - bx, ay - by)
            if d < best_dist:
                best_dist = d
                best_b = (idx, b)
        if best_b is not None and best_dist <= tolerance_px:
            ptsB_used.add(best_b[0])
            matched_pairs.append((a, best_b[1]))

    # Score: matched pairs / average number of points * 100
    avg_count = (len(ptsA) + len(ptsB)) / 2.0
    score = (len(matched_pairs) / avg_count) * 100.0 if avg_count > 0 else 0.0
    return matched_pairs, score

# ----------------------------
# Visualization utilities
# ----------------------------
def draw_minutiae_overlay(img: np.ndarray, endings: List[Point], bifurcations: List[Point]) -> np.ndarray:
    """
    Draw circles for minutiae on a copy of the image.
    Endings -> red circles
    Bifurcations -> blue circles
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for x, y in endings:
        cv2.circle(out, (x, y), radius=4, color=(0, 0, 255), thickness=2)  # red
    for x, y in bifurcations:
        cv2.circle(out, (x, y), radius=4, color=(255, 0, 0), thickness=2)  # blue
    return out

def draw_match_lines(imgA: np.ndarray, imgB: np.ndarray, pairs: List[Tuple[Point, Point]]) -> np.ndarray:
    """
    Create a combined image (side-by-side) and draw lines connecting matched points.
    Lines are drawn across the seam (green).
    """
    # ensure color
    if imgA.ndim == 2:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
    if imgB.ndim == 2:
        imgB = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    # resize B to match height of A if needed
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    if hA != hB:
        scale = hA / hB
        imgB = cv2.resize(imgB, (int(wB * scale), hA))
        wB = imgB.shape[1]

    combined = np.hstack([imgA, imgB])
    offsetB = wA  # x offset of image B in combined image

    for (ax, ay), (bx, by) in pairs:
        bx_shifted = bx + offsetB
        cv2.line(combined, (ax, ay), (bx_shifted, by), color=(0, 255, 0), thickness=1)
        # draw small circles for visibility
        cv2.circle(combined, (ax, ay), 3, (0, 255, 0), -1)
        cv2.circle(combined, (bx_shifted, by), 3, (0, 255, 0), -1)

    return combined

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fingerprint Recognition", layout="wide", initial_sidebar_state="expanded")

st.title("🔬 Fingerprint Recognition")
st.write(
    "Upload two fingerprint images (reference and probe)."
    " The app will extract minutiae (ridge endings & bifurcations) and compute a simple matching score."
)

# Sidebar controls
st.sidebar.header("Preprocessing & Matching Settings")
use_clahe = st.sidebar.checkbox("Apply CLAHE (contrast enhancement)", value=True)
apply_gabor = st.sidebar.checkbox("Apply Gabor filter (single orientation)", value=False)
gabor_theta = st.sidebar.slider("Gabor orientation (degrees)", min_value=0, max_value=180, value=0)
tolerance_px = st.sidebar.slider("Matching tolerance (pixels)", min_value=5, max_value=40, value=15)
score_threshold = st.sidebar.slider("Decision threshold (%)", min_value=0, max_value=100, value=40)
show_skeletons = st.sidebar.checkbox("Show skeleton images (for debugging)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:** This is an educational implementation. Real systems use robust alignment, "
                    "orientation field matching, and advanced minutiae descriptors.")

# File upload
col1, col2 = st.columns(2)
with col1:
    file_ref = st.file_uploader("Upload Reference Fingerprint", type=["jpg", "png", "bmp", "tif"], key="ref")
with col2:
    file_probe = st.file_uploader("Upload Probe/Test Fingerprint", type=["jpg", "png", "bmp", "tif"], key="probe")

if not (file_ref and file_probe):
    st.info("Upload both images to run matching. Tip: use clear, centered fingerprint images for best results.")
    st.stop()

# read images
def read_image_from_upload(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.read()
    arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # maybe grayscale
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img

img_ref_color = read_image_from_upload(file_ref)
img_probe_color = read_image_from_upload(file_probe)

# Keep grayscale copies for processing
gray_ref = to_gray(img_ref_color)
gray_probe = to_gray(img_probe_color)

# Preprocessing pipeline
def preprocess_pipeline(gray_img: np.ndarray, use_clahe_flag: bool, use_gabor_flag: bool, gabor_theta_deg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gray_enhanced, binary_01, skeleton_01)"""
    processed = gray_img.copy()
    if use_clahe_flag:
        processed = apply_clahe(processed)
    if use_gabor_flag:
        theta_rad = math.radians(gabor_theta_deg)
        processed = gabor_enhance(processed, theta=theta_rad)
    binary = binarize_image(processed)
    skel = skeletonize_image(binary)
    return processed, binary, skel

proc_ref, binary_ref, skel_ref = preprocess_pipeline(gray_ref, use_clahe, apply_gabor, gabor_theta)
proc_probe, binary_probe, skel_probe = preprocess_pipeline(gray_probe, use_clahe, apply_gabor, gabor_theta)

# Extract minutiae
end_ref, bif_ref = get_minutiae_from_skeleton(skel_ref)
end_probe, bif_probe = get_minutiae_from_skeleton(skel_probe)

# Matching endings and bifurcations separately, then combine matched pairs
matched_endings, score_end = match_minutiae(end_ref, end_probe, tolerance_px)
matched_bifs, score_bif = match_minutiae(bif_ref, bif_probe, tolerance_px)

# Combine matched pairs for visual lines: endings + bifurcations
all_matched_pairs = matched_endings + matched_bifs

# Simple final score as weighted average (equal weight here)
final_score = (score_end + score_bif) / 2.0

# Decision
is_match = final_score >= score_threshold

# Visualization (annotated images)
annot_ref = draw_minutiae_overlay(img_ref_color if img_ref_color is not None else gray_ref, end_ref, bif_ref)
annot_probe = draw_minutiae_overlay(img_probe_color if img_probe_color is not None else gray_probe, end_probe, bif_probe)
match_visual = draw_match_lines(annot_ref, annot_probe, all_matched_pairs)

# Display results
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Reference (annotated)")
    st.image(cv2.cvtColor(annot_ref, cv2.COLOR_BGR2RGB), use_column_width=True)
with colB:
    st.subheader("Probe / Test (annotated)")
    st.image(cv2.cvtColor(annot_probe, cv2.COLOR_BGR2RGB), use_column_width=True)

st.markdown("---")
left, right = st.columns([1, 1])
with left:
    st.metric("Matching Score (%)", f"{final_score:.2f}")
    st.write(f"Endings matched score: {score_end:.2f}% — matched pairs: {len(matched_endings)}")
    st.write(f"Bifurcations matched score: {score_bif:.2f}% — matched pairs: {len(matched_bifs)}")
    st.write("Decision threshold:", f"{score_threshold}%")
    if is_match:
        st.success("✅ Fingerprints Match (according to this simple algorithm)")
    else:
        st.error("❌ No match (score below threshold)")

with right:
    st.subheader("Matched pairs visualization")
    st.image(cv2.cvtColor(match_visual, cv2.COLOR_BGR2RGB), use_column_width=True)

if show_skeletons:
    st.markdown("**Skeleton images (debug)**")
    s1, s2 = st.columns(2)
    s1.image(skel_ref * 255, caption="Reference skeleton", use_column_width=True)
    s2.image(skel_probe * 255, caption="Probe skeleton", use_column_width=True)

# Allow user to download the combined annotated match image
buf = cv2.imencode(".png", cv2.cvtColor(match_visual, cv2.COLOR_BGR2RGB))[1].tobytes()
st.download_button(label="Download annotated match image (PNG)", data=buf, file_name="fingerprint_match.png", mime="image/png")

# Helpful summary for your assignment (copyable)
st.markdown("---")
st.subheader("Write-up")
st.markdown(
    """
- **Preprocessing:** Images were converted to grayscale, optionally enhanced with CLAHE, optionally filtered with a Gabor kernel to highlight ridge patterns, then binarized using Otsu thresholding.
- **Skeletonization:** Binarized ridges were skeletonized to single-pixel-wide ridge lines using `skimage.morphology.skeletonize`.
- **Minutiae Extraction:** Ridge endings and bifurcations were detected using a simple neighbourhood-count method:
  - ridge ending: pixel with exactly 1 neighbour
  - bifurcation: pixel with exactly 3 neighbours
- **Matching:** Each minutiae point from the reference image was matched to the nearest minutiae in the probe image within a pixel tolerance. Score is percentage of matched points relative to average point count.
- **Limitations:** This is an academic approach useful for demonstration. Production systems require robust alignment, orientation-field matching, ridge orientation/quality checks, and better descriptors (or ML based methods).
"""
)

st.caption("If you want, I can: (a) add rotation alignment, (b) compute orientation angles for minutiae, or (c) convert matching to use a descriptor-based matching method. Tell me which one you want next.")
