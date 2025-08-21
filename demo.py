import cv2
import numpy as np
from core import ImageDehazer

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 320

def normalize_to_uint8(img):
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

    # Create window, start in fullscreen
    cv2.namedWindow("Dehazed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Dehazed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True  # default state

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        dehazed_img, _ = dehazer.dehaze(frame)

        # Normalize and resize to fit 480x320 screen
        dehazed_display = normalize_to_uint8(dehazed_img)
        resized_dehazed = cv2.resize(dehazed_display, (SCREEN_WIDTH, SCREEN_HEIGHT))

        cv2.imshow("Dehazed", resized_dehazed)

        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):   # Quit
            break
        elif key == ord('f'):  # Toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("Dehazed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("🔲 Entered fullscreen mode")
            else:
                cv2.setWindowProperty("Dehazed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("🔳 Exited fullscreen mode")

    cap.release()
    cv2.destroyAllWindows()
