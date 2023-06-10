import cv2

import tensorflow as tf
import numpy as npy
face = cv2.CascadeClassifier(
    "/haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model('./model/model.h5')

labels = ['angry', 'disgusted', 'fearful',
          'happy', 'neutral', 'sad', 'surprised']


def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1, 48, 48, 3))
    print(y_pred)
    label_prep = npy.argmax(y_pred, axis=1)
    print(label_prep)

    return (labels[label_prep[0]])


# sam1 = cv2.imread('./fer2013/test/fearful/im0.png')
# sam1 = cv2.resize(sam1, (48, 48))
# y = detect_face_mask(sam1)
# print(y)


def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, cv2.FILLED)
    end_x = pos[0]+text_size[0][0]+2
    end_y = pos[1]+text_size[0][1]-2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.resize(frame, (48, 48))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    y_pred = detect_face_mask(img)
    print(y_pred)
    draw_label(frame, y_pred, (30, 30), (0, 0, 255))

    # detect faces
    faces = face.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
