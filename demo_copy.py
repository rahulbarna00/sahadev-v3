import cv2
import numpy as np
from core import ImageDehazer

def resize_to_same_height(images, target_height=360):
    """
    Resizes all images to the same target height while preserving aspect ratio.
    """
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized)
    return resized_images

def normalize_to_uint8(img):
    """
    Converts images to uint8 for display:
    - Normalizes float or 1-channel maps.
    - Converts grayscale to BGR for consistency.
    """
    if len(img.shape) == 2:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)
    return img

if __name__ == "__main__":
    video_source = "video7.mp4"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30

    dehazer = ImageDehazer()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        # Dehazing
        dehazed_img, transmission_map = dehazer.dehaze(frame)

        # Normalize
        input_display = normalize_to_uint8(frame)
        dehazed_display = normalize_to_uint8(dehazed_img)
        transmission_map_display = normalize_to_uint8(transmission_map)

        # Resize
        resized_dehazed = resize_to_same_height([dehazed_display], target_height=360)[0]
        resized_aux = resize_to_same_height(
            [input_display, transmission_map_display],
            target_height=360
        )

        # cv2.namedWindow("Dehazed", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Dehazed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("Dehazed", resized_dehazed)

        # Display
        cv2.imshow("Dehazed", resized_dehazed)
        # cv2.imshow("Original | Transmission Map", cv2.hconcat(resized_aux))

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
