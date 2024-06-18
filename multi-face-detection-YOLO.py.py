import cv2
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-face.pt')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #edit the resolution based on your available webcam for ideal results. This is for 720p.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        results = model.track(source=frame, tracker='botsort.yaml', persist=True)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = box.id if hasattr(box, 'id') else None  

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if track_id is not None:
                        cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


