from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

# 初始化Flask应用
app = Flask(__name__)

# 设置上传文件的目录
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 设置最大上传文件大小为16MB

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 加载预训练的MobileNetV2模型
model = MobileNetV2(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def is_cat_or_dog(predictions):
    cat_keywords = [
        'tabby', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat', 'tiger_cat', 
        'cougar', 'lynx', 'leopard', 'cheetah', 'jaguar', 'lion', 'panther'
    ]
    dog_keywords = [
        'Labrador_retriever', 'German_shepherd', 'golden_retriever', 'beagle',
        'bulldog', 'poodle', 'Siberian_husky', 'Chihuahua', 'boxer', 'Dachshund',
        'Pomeranian', 'Shih-Tzu', 'Cocker_spaniel', 'Border_collie', 'Great_Dane',
        'Doberman', 'Rottweiler', 'French_bulldog', 'Pug', 'Maltese_dog', 'Akita',
        'Bernese_mountain_dog', 'Corgi', 'Australian_shepherd', 'Dalmatian', 
        'Schnauzer', 'Shiba_Inu', 'Saint_Bernard', 'Bichon_Frise'
    ]

    for imagenet_id, label, score in predictions:
        if label in cat_keywords:
            return 'Cat'
        elif label in dog_keywords:
            return 'Dog'
    return 'Neither'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predictions = classify_image(filepath)
            result = is_cat_or_dog(predictions)
            return render_template('result.html', filename=filename, result=result, predictions=predictions)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
