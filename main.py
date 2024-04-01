import cv2

face_cascade = cv2.CascadeClassifier('faces.xml')
eye_cascade = cv2.CascadeClassifier('eyes.xml')

webcam = cv2.VideoCapture(0)
webcam.set(3, 300)
webcam.set(4, 500)

while True:
    success, img = webcam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:  # сначала нахожу лицо
        # cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), thickness=2) # убираю лишнюю отрисовку
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=2, minNeighbors=8)

        for (ex, ey, ew, eh) in eyes:  # затем на лице нахожу глаза
            # cv2.rectangle(roi_color, (0, ey), (roi_color.shape[0], ey + eh), (0, 0, 0), 1) # убираю лишнюю отрисовку
            roi_color2 = roi_color[ey:ey + eh, 0:roi_color.shape[0]]
            roi_color2 = cv2.GaussianBlur(roi_color2, (51, 51), 51)
            roi_color[ey:ey + eh, 0:roi_color.shape[0]] = roi_color2

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.imwrite('sсreen_1.jpg', img)  # сохраняю закрытием окна
