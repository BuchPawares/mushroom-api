import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# โหลดโมเดล
model = load_model("mushroom_cnn_model.h5")

# class labels ตามที่ใช้ใน API
class_names = ['Not_dangerous', 'Danger']  # ปรับให้ตรงกับโมเดลคุณ

# เลือกรูปภาพที่จะใช้ทดสอบ
image_path = "test_mushroom.jpg"  # ใส่ชื่อไฟล์ที่อยู่ในโฟลเดอร์เดียวกัน

# อ่านและเตรียมรูปภาพ
img = Image.open(image_path).convert('RGB')
img = img.resize((128, 128))  # ปรับให้ตรงกับ input shape ของโมเดล
img_array = np.array(img) / 255.0  # normalize
img_array = np.expand_dims(img_array, axis=0)  # เพิ่ม batch dimension

# ทำนาย
prediction = model.predict(img_array, verbose=0)[0]
predicted_index = np.argmax(prediction)
predicted_label = class_names[predicted_index]
confidence = float(prediction[predicted_index])

# แสดงผล
print("Raw prediction:", prediction)
print("Predicted label:", predicted_label)
print("Confidence:", round(confidence, 4))
