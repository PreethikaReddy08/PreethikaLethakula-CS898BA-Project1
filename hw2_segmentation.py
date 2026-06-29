from pathlib import Path
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("input/HW1_IMG_CS898BA.png")
OUT_DIR = Path("output/hw2")
README_IMG_DIR = Path("README_images")
REFERENCE_MASK_PATH = OUT_DIR / "reference_mask.png"


def make_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    README_IMG_DIR.mkdir(parents=True, exist_ok=True)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read {path}. Check if your image exists inside the input folder."
        )

    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def normalize_image(image):
    b, g, r = cv2.split(image)

    b_equalized = cv2.equalizeHist(b)
    g_equalized = cv2.equalizeHist(g)
    r_equalized = cv2.equalizeHist(r)

    normalized = cv2.merge([b_equalized, g_equalized, r_equalized])

    return normalized


def clean_mask(mask):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if number_of_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return binary


def mask_score(mask):
    binary = mask > 0
    height, width = binary.shape

    area_fraction = binary.mean()

    if area_fraction < 0.005 or area_fraction > 0.90:
        return -999

    y_values, x_values = np.where(binary)

    if len(x_values) == 0:
        return -999

    center_x = width / 2
    center_y = height / 2

    centroid_x = x_values.mean()
    centroid_y = y_values.mean()

    distance = np.sqrt(
        (centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2
    )

    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    centrality = 1 - (distance / max_distance)

    center_crop = binary[
        height // 4 : 3 * height // 4,
        width // 4 : 3 * width // 4
    ]

    center_overlap = center_crop.mean()

    area_score = 1 - abs(area_fraction - 0.25)

    return 2.0 * centrality + 1.5 * center_overlap + area_score


def choose_foreground(raw_mask):
    option_1 = clean_mask(raw_mask)
    option_2 = clean_mask(255 - raw_mask)

    if mask_score(option_2) > mask_score(option_1):
        return option_2

    return option_1


def apply_mask(image, mask):
    foreground = image.copy()
    foreground[mask == 0] = 0

    return foreground


def otsu_segmentation(normalized):
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    _, raw_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return choose_foreground(raw_mask)


def adaptive_segmentation(normalized):
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    raw_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2
    )

    return choose_foreground(raw_mask)


def kmeans_segmentation(normalized):
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

    height, width = hsv.shape[:2]

    pixels = hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    cv2.setRNGSeed(42)

    best_mask = None
    best_k = None
    best_score = -999

    for k in [3, 4, 5]:
        compactness, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )

        labels_2d = labels.reshape((height, width))

        best_mask_for_k = None
        best_score_for_k = -999

        for cluster_id in range(k):
            raw_mask = np.where(labels_2d == cluster_id, 255, 0).astype(np.uint8)
            cleaned = clean_mask(raw_mask)
            score = mask_score(cleaned)

            if score > best_score_for_k:
                best_score_for_k = score
                best_mask_for_k = cleaned

        save_image(OUT_DIR / f"kmeans_k{k}_mask.png", best_mask_for_k)
        save_image(
            OUT_DIR / f"kmeans_k{k}_foreground.png",
            apply_mask(normalized, best_mask_for_k)
        )

        if best_score_for_k > best_score:
            best_score = best_score_for_k
            best_mask = best_mask_for_k
            best_k = k

    return best_mask, best_k


def grabcut_segmentation(normalized):
    height, width = normalized.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    x = int(width * 0.42)
    y = int(height * 0.12)
    rectangle_width = int(width * 0.24)
    rectangle_height = int(height * 0.78)

    rectangle = (x, y, rectangle_width, rectangle_height)

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        normalized,
        mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    grabcut_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return clean_mask(grabcut_mask)


def load_reference_mask(image_shape):
    if not REFERENCE_MASK_PATH.exists():
        print("Reference mask not found yet.")
        print("Next step: run python3 make_reference_mask.py")
        return None

    reference = cv2.imread(str(REFERENCE_MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if reference is None:
        raise RuntimeError("Could not read reference mask.")

    height, width = image_shape

    if reference.shape[:2] != (height, width):
        reference = cv2.resize(
            reference,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    _, reference = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)

    return reference


def calculate_metrics(predicted_mask, reference_mask):
    predicted = predicted_mask > 0
    reference = reference_mask > 0

    intersection = np.logical_and(predicted, reference).sum()
    union = np.logical_or(predicted, reference).sum()

    predicted_area = predicted.sum()
    reference_area = reference.sum()

    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (predicted_area + reference_area) if predicted_area + reference_area != 0 else 0

    return iou, dice


def save_metrics(masks, reference):
    metrics_path = OUT_DIR / "metrics.csv"

    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Method",
            "IoU_Jaccard_Index",
            "Dice_Coefficient"
        ])

        for method_name, mask in masks.items():
            iou, dice = calculate_metrics(mask, reference)

            writer.writerow([
                method_name,
                f"{iou:.4f}",
                f"{dice:.4f}"
            ])

    print(f"Saved metrics to {metrics_path}")


def create_plot(original, normalized, masks):
    items = [
        ("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)),
        ("Normalized", cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    ]

    for method_name, mask in masks.items():
        items.append((method_name, mask))

    figure, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))

    for axis, item in zip(axes, items):
        title, image = item

        if len(image.shape) == 2:
            axis.imshow(image, cmap="gray")
        else:
            axis.imshow(image)

        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()

    plot_path = README_IMG_DIR / "hw2_comparison.png"

    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to {plot_path}")


def main():
    make_dirs()

    original = read_image(INPUT_PATH)

    normalized = normalize_image(original)

    save_image(OUT_DIR / "original.png", original)
    save_image(OUT_DIR / "normalized_equalized_color.png", normalized)

    otsu_mask = otsu_segmentation(normalized)
    adaptive_mask = adaptive_segmentation(normalized)
    kmeans_mask, chosen_k = kmeans_segmentation(normalized)
    grabcut_mask = grabcut_segmentation(normalized)

    masks = {
        "Otsu": otsu_mask,
        "Adaptive": adaptive_mask,
        f"K-Means K={chosen_k}": kmeans_mask,
        "GrabCut": grabcut_mask
    }

    for method_name, mask in masks.items():
        clean_name = (
            method_name.lower()
            .replace(" ", "_")
            .replace("-", "")
            .replace("=", "")
        )

        save_image(OUT_DIR / f"{clean_name}_mask.png", mask)
        save_image(
            OUT_DIR / f"{clean_name}_foreground.png",
            apply_mask(normalized, mask)
        )

    create_plot(original, normalized, masks)

    reference = load_reference_mask(original.shape[:2])

    if reference is not None:
        save_metrics(masks, reference)

    print("Homework Two segmentation script finished.")
    print(f"Chosen K-Means value: K={chosen_k}")


if __name__ == "__main__":
    main()
