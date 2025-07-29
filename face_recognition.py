import cv2
import os

# تحميل النموذج المدرب
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# تحميل كاشف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# اسم الشخص اللي دربنا عليه
names = {1: "Adriana"}

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if confidence < 80:  # كل ما قل الرقم زادت الدقة
            name = names.get(face_id, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
