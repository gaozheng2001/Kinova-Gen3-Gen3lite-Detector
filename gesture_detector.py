import math
import cv2
import numpy as np
import mediapipe as mp
import time

import pyrealsense2 as rs

def finger_check(landmark):

    if len(landmark) < 21:
        return 'bad detection'
    
    #wriet = landmark[0]
    thumb = landmark[1:5]
    index = landmark[5:9]
    middle = landmark[9:13]
    ring = landmark[13:17]
    pinky = landmark[17:]

    str_fingers = [straight_finger(thumb), straight_finger(index), straight_finger(middle), straight_finger(ring), straight_finger(pinky)]
    print(str_fingers)
    if is_straight_finger(str_fingers, [1]):
        if finger_up(index):
            return 'up'

    return ''

def straight_finger(finger) :
    error_angle = 10

    mcp, pip, dip, tip = finger[0], finger[1], finger[2], finger[3]

    k1x, k1y = (mcp.x - pip.x)/distance(mcp, pip), (mcp.y - pip.y)/distance(mcp, pip)
    k2x, k2y = (pip.x - dip.x)/distance(pip, dip), (pip.y - dip.y)/distance(pip, dip)
    k3x, k3y = (dip.x - tip.x)/distance(dip, tip), (dip.y - tip.y)/distance(dip, tip)
    
    if abs(k1x*k2y - k2x*k1y) < math.sin(math.pi*error_angle/180) and abs(k3x*k2y - k2x*k3y) < math.sin(math.pi*error_angle/180):
        return True
    else:
        return False

def is_straight_finger(fingers, require):
    result = 0
    all = [0,1,2,3,4]
    unrequire = list(set(all) - set(require))
    for index in range(len(require)):
        if fingers[require[index]]:
            result += 1
    for index in range(len(unrequire)):
        if fingers[unrequire[index]]:
            result -= 1
    return result == len(require)


def finger_up(finger):
    error_angle = 10

    mcp,tip = finger[0], finger[3]
    ky = (tip.y - mcp.y)/distance(mcp, tip)
    if -ky > math.sin(math.pi*(1/2 - error_angle/180)):
        return True
    return False

def closer_point(landmark):
    min = landmark[0].z
    for point in landmark:
        if min > point.z:
            min = point.z
    return min

def farr_point(landmark):
    max = landmark[0].z
    for point in landmark:
        if max < point.z:
            max = point.z
    return max

def distance(point1, point2):
    return math.sqrt(math.pow(point1.x - point2.x, 2) 
                + math.pow(point1.y - point2.y, 2))

if __name__ == '__main__':

    # RGB and dpeth strams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    
    # Get depth camera's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print('Depth Scale is: ' + str(depth_scale))

    # Align depth frame to RGB frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 250)
    sensor.set_option(rs.option.auto_exposure_priority, True)

    # grasp drawing params
    radius = 1
    color = (0, 0, 255)
    thickness = 2
    isClosed = True

    #cap = cv2.VideoCapture('rtsp://192.168.1.10/color')
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    
    pTime = 0
    cTime = 0

    detected = 0
    
    while True:
        detected += 1

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        #success, img = cap.read()
        img = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                str_fingers = finger_check(handLms.landmark)
                #print('new check')
                #print(str_fingers)
                #print(handLms.landmark[0].z)
                #print(img.shape[1]*farr_point(handLms.landmark))
                #print(img.shape[1]*closer_point(handLms.landmark))
                if detected%15 == 0:
                    print('30fpes')
                    

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    
                    cv2.circle(img, (cx,cy), 3, (255, 0, 255), cv2.FILLED)
    
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
        cv2.imshow("Image", img)
    
        if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    #cap.release()
    cv2.destroyAllWindows()

