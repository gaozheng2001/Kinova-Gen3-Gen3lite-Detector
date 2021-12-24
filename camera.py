"""Terminal command:

color: gst-launch-1.0 rtspsrc location=rtsp://192.168.1.10/color latency=30 ! rtph264depay ! avdec_h264 ! autovideosink

depth: gst-launch-1.0 rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtpgstdepay ! videoconvert ! autovideosink
"""

import numpy as np
import cv2
import dlib
# from imutils import face_utils
import time

def gamma_trans(img,gamma):#gamma函数处理
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#建立映射表
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#颜色值为整数
    return cv2.LUT(img,gamma_table)

if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture("rtsp://192.168.1.10/color")
    #video_capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    FACTOR =  0.5


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        start = time.time()

        res = cv2.resize(frame, None, fx=FACTOR, fy=FACTOR)
        #Gets a grayscale image
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        rects = detector(gray)    

        #cv2.imshow('GRAY', gray)
        #调整曝光
        gramed = gamma_trans(frame, 0.5)


        #haar_cascade_face = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')
        #faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

        #for (x,y,w,h) in faces_rects:
        #  cv2.rectangle(gramed, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        for (i, rect) in enumerate(rects):
            x1 = int(rect.left() / FACTOR)
            y1 = int(rect.top() / FACTOR)
            x2 = int(rect.right() / FACTOR)
            y2 = int(rect.bottom() / FACTOR)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(gramed, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the video output
        #cv2.imshow('Video', frame)
        cv2.imshow('Video', gramed )

        # Quit video by typing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()

        print(chr(27) + "[2J")
        print("FPS: {}\nFaces: {}".format(1/(end - start), len(rects)))


    video_capture.release()
    cv2.destroyAllWindows()

