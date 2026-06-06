import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector_model.h5")

# Start webcam
camera = cv2.VideoCapture(0)

print("Face Mask Detection Started...")

while True:
    success, frame = camera.read()

    if not success:
        break

    # Resize image for prediction
    image = cv2.resize(frame, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image, verbose=0)[0][0]

    if prediction > 0.5:
        label = "Mask Detected"
    else:
        label = "No Mask"

    # Show result
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Mask Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()