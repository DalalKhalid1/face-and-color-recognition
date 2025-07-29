import cv2
import os
import numpy as np

# مسار الصور
dataset_path = "C:/Users/dalal/Desktop/dataset/adriana"


# نجهز القوائم للصور والـ IDs
faces = []
labels = []

# نحدد ID للشخص
person_id = 1

# نقرأ كل الصور بالفولدر
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        continue
    
    img = cv2.resize(img, (200, 200))
    
    faces.append(img)
    labels.append(person_id)

# نحول الليستة إلى numpy array
faces = np.array(faces)
labels = np.array(labels)

# نستخدم LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# نحفظ النموذج
os.makedirs("trainer", exist_ok=True)
recognizer.save("C:/Users/dalal/Desktop/trainer/trainer.yml")

print("Training complete! File saved as trainer/trainer.yml")
 
 