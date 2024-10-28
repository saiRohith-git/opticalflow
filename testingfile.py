import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
else:
    print("Camera is working.")
    cap.release()
print("erri;papadsfj")