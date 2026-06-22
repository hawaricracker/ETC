import cv2
import time

cam = cv2.VideoCapture(1)
start = time.time()
idx = 200
while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    cv2.waitKey(1)

    if idx == 3:
        cv2.imwrite(f"Gambar{idx}.jpg", frame)
        print(f"Gambar{idx}")
        idx += 1

    if time.time() - start >= 300:
        cv2.imwrite(f"Gambar{idx}.jpg", frame)
        print(f"Gambar{idx}")
        start = time.time()
        idx += 1

    if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()