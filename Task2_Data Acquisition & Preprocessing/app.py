import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from PIL import Image

# 1. [Flask Tool & Step 5]: 
app = Flask(__name__)

# 2. [YOLOv8 Tool & Step 2]: 
print("[INFO] Loading YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt")

# [Step 4: Optimization]: 
if not os.path.exists("yolov8n_saved_model") and not os.path.exists("yolov8n.tflite"):
    print("[INFO] Optimizing model for faster performance...")
    try:
        yolo_model.export(format="tflite")
    except Exception as e:
        print(f"[INFO] Optimization complete or skipped: {e}")

# 3. [TensorFlow/Keras Tool]: 
print("[INFO] Loading TensorFlow/Keras Deep Analysis Model...")
keras_model = tf.keras.applications.MobileNetV2(weights="imagenet")

latest_frame_for_tf = None


# 4. [OpenCV Tool & Step 1, 3]: 
def generate_frames():
    global latest_frame_for_tf
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam stream started successfully.")

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # TensorFlow 
        latest_frame_for_tf = frame.copy()

        # YOLOv8 
        results = yolo_model(frame, verbose=False)
        
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


# 5. [Flask Deployment]: 
@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
   
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_frame')
def analyze_frame():
    
    global latest_frame_for_tf
    if latest_frame_for_tf is None:
        return jsonify({"error": "No image captured"})
 
    rgb_image = cv2.cvtColor(latest_frame_for_tf, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image).resize((224, 224))
    img_array = np.array(pil_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # TensorFlow/Keras 
    predictions = keras_model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    
   
    _, object_label, confidence_score = decoded[0]
    percentage = f"{confidence_score * 100:.2f}%"

    
    return jsonify({
        "label": object_label.replace("_", " ").title(),
        "confidence": percentage
    })

if __name__ == '__main__':
    print("[INFO] Launching local web server on http://127.0.0.1:5000/")
    app.run(debug=False, port=5000)