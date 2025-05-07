from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# โหลดโมเดล
model = load_model("mushroom_cnn_model.h5")
class_names = ['ระโงก', 'ระงาก'] # เปลี่ยนตามโมเดลคุณ

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert("RGB").resize((128, 128))  # ขนาดตรงกับโมเดล
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({"class": predicted_class})

# สำหรับโฮสต์บน Render ต้องระบุ host และ port
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
