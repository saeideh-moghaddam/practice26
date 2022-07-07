import cv2
import numpy as np

video_cap = cv2.VideoCapture(0)
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    color_faded = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width, height = color_faded.shape
    
    goal = color_faded[(width//8)*3:(width//8)*5, (height//8)*3:(height//8)*5]

    filter_camera = np.ones((45, 45), np.float32)/2025
    color_faded = cv2.filter2D(color_faded, -1, filter_camera)
    al = 3 
    be = 0 
    enhanced_goale = cv2.convertScaleAbs(goal, alpha=al, beta=be)

    color_faded[(width//8)*3:(width//8)*5, (height//8)*3:(height//8)*5] = enhanced_goale
    cv2.rectangle(color_faded, (height//8*3, width//8*3), ((height//8*5), (width//8*5)), (0, 0, 0), 4)

    target_color = np.average(enhanced_goale)
    if target_color<85:
        cv2.putText(color_faded, "Black", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
    elif target_color<170:
        cv2.putText(color_faded, "Gray", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (127,127,127))
    else:
        cv2.putText(color_faded, "White", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

    re = cv2.VideoWriter()
    re.write(color_faded)
    cv2.imshow("faded", color_faded)
    re.release()
    cv2.waitKey(5)
       


