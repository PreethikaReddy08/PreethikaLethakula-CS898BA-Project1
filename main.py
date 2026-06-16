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