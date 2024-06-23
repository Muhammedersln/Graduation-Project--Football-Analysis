from ultralytics import YOLO

model = YOLO('models_yolov5/weights/best.pt')

results= model.predict('input_videos/121364_3.mp4',save=True)

print(results[0])

for box  in results[0].boxes:
    print(box)
