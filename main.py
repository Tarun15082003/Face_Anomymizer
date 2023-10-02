import cv2
import mediapipe as mp

# read image
cap = cv2.VideoCapture(0)

# detect face
mp_face_detection = mp.solutions.face_detection

while True:
    ret, img = cap.read()

    H, W, _ = img.shape
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)
        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                x1 = int(x1*W)
                y1 = int(y1*H)
                w = int(w*W)
                h = int(h*H)

                img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (100, 100))

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



