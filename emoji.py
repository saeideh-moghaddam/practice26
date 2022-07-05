from sys import maxsize
import cv2
import cvzone
import keyboard

detector_sm = cv2.CascadeClassifier("haarcascade_smile.xml")
detector_fa = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detector_ey = cv2.CascadeClassifier("haarcascade_eye.xml")

emogi_s = cv2.imread("lip.png", cv2.IMREAD_UNCHANGED)
emogi_f = cv2.imread("emogi.png", cv2.IMREAD_UNCHANGED)
emogi_e = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)

video_capture = cv2.VideoCapture(0)
while True:    
    ret, frame = video_capture.read()
    if ret == False:
        break

    k = cv2.waitKey(1)
    if keyboard.is_pressed('1'):
        FACES = detector_fa.detectMultiScale(frame, 1.3)

        for (x, y, w, h) in FACES:
            finalEmojy = cv2.resize(emogi_f, (w, h))
            frame = cvzone.overlayPNG(frame, finalEmojy, [x, y])

    if keyboard.is_pressed('2'):
        EYE = detector_ey.detectMultiScale(frame, 2, maxSize=(50,50))

        for (x, y, w, h) in EYE:
            finalEmojy = cv2.resize(emogi_e, (w, h))
            frame = cvzone.overlayPNG(frame, finalEmojy, [x, y])  

        LIP = detector_sm.detectMultiScale(frame, 1.3, 15)
        for (x, y, w, h) in LIP:
            finalEmojy = cv2.resize(emogi_s, (w, h))
            frame = cvzone.overlayPNG(frame, finalEmojy, [x, y])
            
    if keyboard.is_pressed('3'):
        FACES = detector_fa.detectMultiScale(frame, 1.3)

        for (x, y, w, h) in FACES:            
            blurred = frame[y:y+h, x:x+w]
            pixlated = cv2.resize(blurred, (15, 15), interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(pixlated, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = output

     
    if keyboard.is_pressed('4'):
        FACES = detector_fa.detectMultiScale(frame, 1.3)
        for (x, y, w, h) in FACES:
            blurred = cv2.GaussianBlur(frame[y:y+h, x:x+w], (25, 25), 35)
            frame[y:y+h, x:x+w] = blurred       
   
    if keyboard.is_pressed('Esc'):
        exit()
    cv2.imshow("'You're beautiful'", frame)