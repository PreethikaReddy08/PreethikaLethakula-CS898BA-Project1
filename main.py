import cv2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew
from pathlib import Path

# File paths
INPUT_IMAGE_PATH = Path("input/HW1_IMG_CS898BA.png")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load image using OpenCV
# OpenCV loads color images in BGR order by default
image_bgr = cv2.imread(str(INPUT_IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(f"Could not load image from {INPUT_IMAGE_PATH}")

# Convert BGR to RGB so the channel names are easier to understand
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

channel_names = ["Red", "Green", "Blue"]
statistics_rows = []

for index, channel_name in enumerate(channel_names):
    channel = image_rgb[:, :, index]
    pixel_values = channel.flatten()

    minimum = np.min(pixel_values)
    maximum = np.max(pixel_values)
    mean = np.mean(pixel_values)
    median = np.median(pixel_values)
    mode = stats.mode(pixel_values, keepdims=True).mode[0]
    skewness = skew(pixel_values)
    pixel_range = maximum - minimum
    standard_deviation = np.std(pixel_values)
    variance = np.var(pixel_values)

    statistics_rows.append({
        "Channel": channel_name,
        "Minimum": minimum,
        "Maximum": maximum,
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Skewness": skewness,
        "Range": pixel_range,
        "Standard Deviation": standard_deviation,
        "Variance": variance
    })

# Convert results to a table
statistics_df = pd.DataFrame(statistics_rows)

# Print results
print("\nImage Statistics by RGB Channel")
print(statistics_df.to_string(index=False))

# Save results to CSV
csv_path = OUTPUT_DIR / "image_statistics.csv"
statistics_df.to_csv(csv_path, index=False)

print(f"\nStatistics saved to: {csv_path}")

# -----------------------------
# Part 2 - Image Conversions
# -----------------------------

# Save original image
cv2.imwrite(str(OUTPUT_DIR / "01_original.png"), image_bgr)

# Grayscale
grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(OUTPUT_DIR / "02_grayscale.png"), grayscale)

# Binary image
_, binary = cv2.threshold(grayscale, 80, 255, cv2.THRESH_BINARY)
cv2.imwrite(str(OUTPUT_DIR / "03_binary.png"), binary)

# HSV
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
cv2.imwrite(str(OUTPUT_DIR / "04_hsv.png"), hsv)

# LAB (CIELAB)
lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
cv2.imwrite(str(OUTPUT_DIR / "05_lab.png"), lab)

# HLS
hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
cv2.imwrite(str(OUTPUT_DIR / "06_hls.png"), hls)

# Histogram Equalization on V channel
hsv_equalized = hsv.copy()
hsv_equalized[:, :, 2] = cv2.equalizeHist(hsv_equalized[:, :, 2])

equalized_bgr = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

cv2.imwrite(
    str(OUTPUT_DIR / "07_hsv_equalized_rgb.png"),
    equalized_bgr
)

print("\nColor conversion images saved successfully.")

# -----------------------------
# Part 2 - Affine Transformations
# -----------------------------

base_image_files = [
    "01_original.png",
    "02_grayscale.png",
    "03_binary.png",
    "04_hsv.png",
    "05_lab.png",
    "06_hls.png",
    "07_hsv_equalized_rgb.png"
]

rotation_angles = [10, 20, 30, 40, 50, 60, 70]

translations = [
    (10, 5),
    (20, 10),
    (30, 15),
    (40, 20),
    (50, 25),
    (60, 30),
    (70, 35)
]

for index, file_name in enumerate(base_image_files):
    image_path = OUTPUT_DIR / file_name
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Could not load {file_name}")
        continue

    height, width = img.shape[:2]

    angle = rotation_angles[index]
    tx, ty = translations[index]

    # Transformation 1: unique rotation
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(str(OUTPUT_DIR / f"{file_name[:-4]}_affine_rotation_{angle}deg.png"), rotated)

    # Transformation 2: unique translation
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    cv2.imwrite(str(OUTPUT_DIR / f"{file_name[:-4]}_affine_translation_{tx}_{ty}.png"), translated)

print("\nUnique affine transformation images saved successfully.")

# -----------------------------
# Part 2 - Gaussian Blur
# -----------------------------

all_image_files = [
    file.name for file in OUTPUT_DIR.glob("*.png")
    if "gaussian" not in file.name
]

sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

for file_name in all_image_files:
    image_path = OUTPUT_DIR / file_name
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Could not load {file_name}")
        continue

    for sigma in sigma_values:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        output_name = f"{file_name[:-4]}_gaussian_sigma_{sigma}.png"
        cv2.imwrite(str(OUTPUT_DIR / output_name), blurred)

print("\nGaussian blur images saved successfully.")

# -----------------------------
# Part 3 - Create Random Subsets
# -----------------------------

import random
import shutil

SUBSET_DIR = Path("subsets")
SUBSET_DIR.mkdir(exist_ok=True)

all_images = list(OUTPUT_DIR.glob("*.png"))

random.seed(42)
random.shuffle(all_images)

subset_size = 42

for i in range(4):
    subset_folder = SUBSET_DIR / f"subset_{i + 1}"
    subset_folder.mkdir(exist_ok=True)

    start_index = i * subset_size
    end_index = start_index + subset_size

    subset_images = all_images[start_index:end_index]

    for image_path in subset_images:
        destination = subset_folder / image_path.name
        shutil.copy(image_path, destination)

print("\nCreated 4 subsets with 42 images each.")

# -----------------------------
# Part 3 - Edge Detection
# -----------------------------

EDGE_OUTPUT_DIR = Path("edge_outputs")
EDGE_OUTPUT_DIR.mkdir(exist_ok=True)

SELECTED_SUBSET = Path("subsets/subset_1")

def convert_to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sobel_edge(img):
    gray = convert_to_gray(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    combined = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(combined)

def laplacian_edge(img):
    gray = convert_to_gray(img)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def canny_edge(img):
    gray = convert_to_gray(img)
    return cv2.Canny(gray, 50, 150)

def prewitt_edge(img):
    gray = convert_to_gray(img)

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    edge_x = cv2.filter2D(gray, -1, kernel_x)
    edge_y = cv2.filter2D(gray, -1, kernel_y)

    return cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)

for image_path in SELECTED_SUBSET.glob("*.png"):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Could not load {image_path.name}")
        continue

    base_name = image_path.stem

    cv2.imwrite(str(EDGE_OUTPUT_DIR / f"{base_name}_input.png"), img)
    cv2.imwrite(str(EDGE_OUTPUT_DIR / f"{base_name}_sobel.png"), sobel_edge(img))
    cv2.imwrite(str(EDGE_OUTPUT_DIR / f"{base_name}_laplacian.png"), laplacian_edge(img))
    cv2.imwrite(str(EDGE_OUTPUT_DIR / f"{base_name}_canny.png"), canny_edge(img))
    cv2.imwrite(str(EDGE_OUTPUT_DIR / f"{base_name}_prewitt.png"), prewitt_edge(img))

print("\nEdge detection images saved successfully.")

# -----------------------------
# Create Comparison Plots
# -----------------------------

import matplotlib.pyplot as plt

COMPARISON_PLOTS_DIR = Path("plots/comparison_plots")
COMPARISON_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

input_files = sorted(EDGE_OUTPUT_DIR.glob("*_input.png"))

for input_file in input_files:
    base_name = input_file.stem.replace("_input", "")

    sobel_file = EDGE_OUTPUT_DIR / f"{base_name}_sobel.png"
    laplacian_file = EDGE_OUTPUT_DIR / f"{base_name}_laplacian.png"
    canny_file = EDGE_OUTPUT_DIR / f"{base_name}_canny.png"
    prewitt_file = EDGE_OUTPUT_DIR / f"{base_name}_prewitt.png"

    input_img = cv2.imread(str(input_file))
    sobel_img = cv2.imread(str(sobel_file), cv2.IMREAD_GRAYSCALE)
    laplacian_img = cv2.imread(str(laplacian_file), cv2.IMREAD_GRAYSCALE)
    canny_img = cv2.imread(str(canny_file), cv2.IMREAD_GRAYSCALE)
    prewitt_img = cv2.imread(str(prewitt_file), cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#1e1e1e")

    plt.suptitle(
        f"Pipeline Trajectory:\n{base_name}",
        color="cyan",
        fontsize=12
    )

    images = [
        (sobel_img, "Sobel Edge", "gray", 2),
        (laplacian_img, "Laplacian Edge", "gray", 4),
        (input_img, "Input Image", None, 5),
        (canny_img, "Canny Edge", "gray", 6),
        (prewitt_img, "Prewitt Edge", "gray", 8),
    ]

    for img, title, cmap, position in images:
        ax = plt.subplot(3, 3, position)
        ax.set_facecolor("#1e1e1e")

        if img is None:
            ax.text(0.5, 0.5, "Image missing", color="white", ha="center")
        elif cmap == "gray":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(title, color="white")
        ax.axis("off")

    plt.tight_layout()

    plot_path = COMPARISON_PLOTS_DIR / f"{base_name}_comparison.png"
    plt.savefig(plot_path, facecolor=fig.get_facecolor())
    plt.close()

print("\nCreated 42 comparison plots successfully.")