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