from pathlib import Path
import cv2
import numpy as np


INPUT_PATH = Path("input/HW1_IMG_CS898BA.png")
OUT_DIR = Path("output/hw2")
REFERENCE_MASK_PATH = OUT_DIR / "reference_mask.png"

TARGET_POINTS = 35

points = []
done = False


def save_mask(image_shape):
    global done

    if len(points) < 3:
        print("Need at least 3 points.")
        return

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    polygon = np.array(points, dtype=np.int32)

    cv2.fillPoly(mask, [polygon], 255)
    cv2.imwrite(str(REFERENCE_MASK_PATH), mask)

    print(f"Saved reference mask to {REFERENCE_MASK_PATH}")
    done = True


def redraw_image(base_image):
    display_image = base_image.copy()

    for point in points:
        cv2.circle(display_image, point, 4, (0, 0, 255), -1)

    if len(points) >= 2:
        cv2.polylines(
            display_image,
            [np.array(points, dtype=np.int32)],
            False,
            (0, 255, 0),
            2
        )

    text1 = f"Click around full figure: {len(points)}/{TARGET_POINTS} points"
    text2 = "It will auto-save after 30 clicks"

    cv2.putText(
        display_image,
        text1,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_image,
        text2,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return display_image


def mouse_callback(event, x, y, flags, param):
    image = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}/{TARGET_POINTS}")

        if len(points) >= TARGET_POINTS:
            save_mask(image.shape)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
            print(f"Undo. Current points: {len(points)}/{TARGET_POINTS}")


def main():
    global done

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read {INPUT_PATH}. Check the image name inside the input folder."
        )

    window_name = "Draw Reference Mask"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, image)

    while not done:
        display_image = redraw_image(image)
        cv2.imshow(window_name, display_image)
        cv2.waitKey(20)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
