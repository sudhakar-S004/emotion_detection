"""
main.py — Real-time Emotion Detection
Run AFTER training is complete (model.h5 must exist).

Usage:
    python main.py

Press Q to quit.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('model.h5')

# ── Load face detector ────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ── Emotion labels — alphabetical order (matches how Keras reads folders) ─────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

COLOURS = {
    'Angry':    (0,   0,   220),
    'Disgust':  (0,   128, 0),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   215, 255),
    'Neutral':  (180, 180, 180),
    'Sad':      (220, 100, 0),
    'Surprise': (0,   200, 200),
}

def predict_emotion(face_roi):
    resized    = cv2.resize(face_roi, (48, 48))
    normalised = resized.reshape(1, 48, 48, 1).astype('float32') / 255.0
    preds      = model.predict(normalised, verbose=0)
    idx        = int(np.argmax(preds))
    return EMOTIONS[idx], float(preds[0][idx])


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam open — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi           = gray[y:y+h, x:x+w]
            emotion, conf = predict_emotion(roi)
            colour        = COLOURS.get(emotion, (0, 255, 0))

            cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(frame, f"{emotion}  {conf*100:.0f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour, 2)

        cv2.imshow('Emotion Detection  (Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()