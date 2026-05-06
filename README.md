# Emotion Detection AI — Project README

## What this project does
Uses your webcam + a trained AI model to detect emotions (Happy, Sad, Angry, etc.)
in real time on every face in the frame.

---

## Step-by-step setup (for beginners)

### Step 1 — Install Python
Download Python 3.8+ from https://python.org and install it.
Make sure to check "Add Python to PATH" during installation.

### Step 2 — Open this folder in VS Code
File → Open Folder → select the emotion_detection/ folder.

### Step 3 — Open a terminal in VS Code
Press Ctrl + ` (backtick key, top-left of keyboard)

### Step 4 — Install libraries
```
pip install -r requirements.txt
```

### Step 5 — Download the FER-2013 dataset
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Create a free Kaggle account if needed
3. Download the dataset (about 63 MB)
4. Extract it so your folder looks like:
```
emotion_detection/
  data/
    train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    test/
      (same sub-folders)
```

### Step 6 — Download the face detector file
1. Go to: https://github.com/opencv/opencv/tree/master/data/haarcascades
2. Download `haarcascade_frontalface_default.xml`
3. Put it in the emotion_detection/ folder (same level as main.py)

### Step 7 — Train the model
```
python train_model.py
```
This will take 20–60 minutes on a normal laptop. If it's too slow, use
Google Colab (free) for GPU training.

When done, a file called `model.h5` will appear in your folder.

### Step 8 — Run the emotion detector!
```
python main.py
```
A webcam window will appear. It draws a coloured box around your face
and shows the detected emotion + confidence percentage.

Press Q to quit.

---

## File overview
| File | Purpose |
|------|---------|
| `main.py` | Runs the real-time webcam emotion detector |
| `train_model.py` | Trains the AI model on the FER-2013 dataset |
| `requirements.txt` | List of Python libraries to install |
| `model.h5` | The saved trained model (created after training) |
| `haarcascade_frontalface_default.xml` | Face detector (download separately) |
| `data/` | Training and test images (download from Kaggle) |

---

## How the AI was trained (simple explanation)

1. **Data**: 35,000+ grayscale face photos (48×48 pixels), each labeled with an emotion
2. **Model**: A Convolutional Neural Network (CNN) with 3 stacked "Conv blocks"
3. **Training**: The model sees a face → guesses the emotion → compares to real label → adjusts its weights → repeats millions of times
4. **Result**: After 50 rounds (epochs), the model gets ~60-65% accurate at identifying emotions

Why not 100%? Emotions are ambiguous — even humans disagree on what face shows which emotion!

---

## Tips & troubleshooting

- **Webcam not opening**: Make sure no other app (Zoom, Teams) is using it
- **Model too slow**: Training on CPU is normal — use Google Colab for speed
- **Low accuracy**: Try more epochs or use a pre-trained model like FER+ 
- **"Module not found"**: Run `pip install -r requirements.txt` again

---

## Want to improve it?

- Use a pre-trained model (DeepFace, FER library) for higher accuracy
- Add a confidence threshold to skip low-confidence detections
- Log detected emotions to a CSV file over time
- Build a simple web app using Flask + this model