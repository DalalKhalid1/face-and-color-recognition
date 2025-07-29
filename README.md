# This repository contains two AI projects:

## 1. Face Recognition
- Detects and recognizes a trained face using OpenCV.
- Includes:
  - `dataset/` → training images
  - `trainer/` → saved model
  - `train_model.py` → training script
  - `face_recognition.py` → live face recognition

## 2. Color Recognition (HuskyLens Simulation)
- Detects red objects using OpenCV.
- Includes:
  - `color_recognition.py` → detects red color via camera
---
![[face-and-color-recognition](https://github.com/DalalKhalid1/face-and-color-recognition)](screenshot2.png)
---

###  How to Run:

1. Install Python and OpenCV.

2. Run face recognition:
   ```bash
   python train_model.py
   python face_recognition.py
   
3. Run color recognition:
   
python color_recognition.py

**Note:**  
- The dataset images are compressed in `dataset.zip`. Unzip it before training or running face recognition.  
- The trained model is compressed in `trainer.zip`. Unzip it before running live recognition.
