from ultralytics import YOLO
model = YOLO('model/best.pt')
results = model.predict(r'C:\Users\hp\Desktop\yolo\08fd33_4.mp4', save=True)
print(results[0])
print("=============================================")
for box in results[0].boxes:
  print(box)
  