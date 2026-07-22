import cv2
import json
import numpy as np

IMAGE_PATH = "parkir.jpg"
OUTPUT_JSON = "parking_slots.json"

img_original = cv2.imread(IMAGE_PATH)

if img_original is None:
    print("Gambar tidak ditemukan!")
    exit()

img = img_original.copy()

slots = []
current_polygon = []

def generate_slot_name(index):

    row = chr(ord('A') + (index // 20))
    number = (index % 20) + 1

    return f"{row}{number}"

def redraw():

    global img

    img = img_original.copy()

    for idx, slot in enumerate(slots):

        pts = np.array(slot["polygon"], np.int32)

        cv2.polylines(
            img,
            [pts],
            True,
            (0,255,0),
            2
        )

        center_x = int(np.mean(pts[:,0]))
        center_y = int(np.mean(pts[:,1]))

        cv2.putText(
            img,
            slot["id"],
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    if len(current_polygon) > 0:

        pts = np.array(current_polygon, np.int32)

        cv2.polylines(
            img,
            [pts],
            False,
            (0,0,255),
            2
        )

        for point in current_polygon:

            cv2.circle(
                img,
                tuple(point),
                5,
                (255,0,0),
                -1
            )

def mouse_callback(event, x, y, flags, param):

    global current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:

        current_polygon.append([x,y])

        print(f"Titik: ({x},{y})")

        redraw()

cv2.namedWindow(
    "Parking Slot Annotation",
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    "Parking Slot Annotation",
    mouse_callback
)

redraw()

print("\n===== PETUNJUK =====")
print("Klik titik polygon")
print("N = Simpan slot")
print("U = Undo titik terakhir")
print("S = Simpan JSON")
print("Q = Keluar")
print("====================\n")

while True:

    cv2.imshow(
        "Parking Slot Annotation",
        img
    )

    key = cv2.waitKey(10) & 0xFF

    if key == ord('n'):

        if len(current_polygon) >= 4:

            slot_name = generate_slot_name(
                len(slots)
            )

            slots.append({
                "id": slot_name,
                "polygon": current_polygon.copy()
            })

            print(
                f"Slot {slot_name} disimpan"
            )

            current_polygon.clear()

            redraw()

        else:

            print(
                "Minimal 4 titik!"
            )

    elif key == ord('u'):

        if len(current_polygon) > 0:

            current_polygon.pop()

            redraw()

    elif key == ord('s'):

        with open(
            OUTPUT_JSON,
            "w"
        ) as f:

            json.dump(
                {"slots": slots},
                f,
                indent=4
            )

        print(
            f"Disimpan ke {OUTPUT_JSON}"
        )

    elif key == ord('q'):

        break

cv2.destroyAllWindows()