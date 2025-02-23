from ultralytics import YOLO
import cv2

model = YOLO(Model Yolu)

cap = cv2.VideoCapture(Video Yolu)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


out = cv2.VideoWriter('kayit1.avi', fourcc, 33.0, (frame_width, frame_height))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow(':)', cv2.WINDOW_NORMAL)
cv2.resizeWindow(':)', frame_width, frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video okunamıyor. Çıkılıyor...")
        break
   
    results = model.predict(frame)
    result = results[0]
    if ret == 1:

        for box in result.boxes:
            class_id = "Sigara"
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]

            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 2)

            # ROI 
            roi = frame[cords[1]: cords[3], cords[0]: cords[2]]
            blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)
            frame[cords[1]:cords[3], cords[0]:cords[2]] = blurred_roi
            cv2.putText(frame, class_id, (cords[0] + 5, cords[3] - 5), font, 1.3, (255, 0, 0), 2)

            out.write(frame)
            cv2.imshow(":)", frame)
            
    else:
        break

    if cv2.waitKey(33) & 0xFF == ord('q'):  
        break

out.release()
cap.release()
cv2.destroyAllWindows()
