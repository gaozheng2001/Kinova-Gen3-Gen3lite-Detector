from typing import List
import numpy as np
import math

import cv2
import dlib
import time
import mediapipe as mp

import pyrealsense2 as rs
from gesture_recognition import gesture

from videocaptureasync import VideoCaptureAsync

from kortex_api.UDPTransport import UDPTransport
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient

from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2


def send_home(base_client_service):
    print('Going Home....')
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base_client_service.ReadAllActions(action_type)
    action_handle = None
    
    for action in action_list.action_list:
        if action.name == "Follow":
            action_handle = action.handle

    base_client_service.ExecuteActionFromReference(action_handle)
    time.sleep(6)
    print(" Done!")

def get_distance(p1, p2):
    return (p2[0]-p1[0], p2[1]-p1[1])

def twist_command(base_client_service, cmd):
    command = Base_pb2.TwistCommand()
    #command.mode = Base_pb2.UNSPECIFIED_TWIST_MODE
    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 2  # Unlimited time to execute

    x, y, z, tx, ty, tz = cmd

    twist = command.twist
    twist.linear_x = x
    twist.linear_y = y
    twist.linear_z = z
    twist.angular_x = tx
    twist.angular_y = ty
    twist.angular_z = tz
    base_client_service.SendTwistCommand(command)

def check_up(finger_state):
    if len(finger_state) < 15:
        return False
    check = 0
    result = False
    for k in finger_state:
        check += 1 if finger_state.get(k) != 'up' else 0
    result = False if check != 0 else True
    return result

if __name__ == "__main__":

    DEVICE_IP = "192.168.1.10"
    DEVICE_PORT = 10000

    # Setup API
    errorCallback = lambda kException: print("_________ callback error _________ {}".format(kException))
    #transport = UDPTransport()
    transport = TCPTransport()
    router = RouterClient(transport, errorCallback)
    transport.connect(DEVICE_IP, DEVICE_PORT)

    # Create session
    print("Creating session for communication")
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = 'admin'
    session_info.password = 'admin'
    session_info.session_inactivity_timeout = 60000   # (milliseconds)
    session_info.connection_inactivity_timeout = 2000 # (milliseconds)
    print("Session created")

    session_manager = SessionManager(router)   
    session_manager.CreateSession(session_info)

    # Create required services
    base_client_service = BaseClient(router)

    send_home(base_client_service)

    #video_capture = VideoCaptureAsync("rtsp://192.168.1.10/color")

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


    detector = dlib.get_frontal_face_detector()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils


    FACTOR = 1
    VEL = -1
    movs = {
        "look_up": np.array((0,VEL,0,0,0,0)),
        "look_left": np.array((VEL,0,0,0,0,0)),
        "look_forword":np.array((0,0,-VEL,0,0,0)),
        "stop": np.array((0,0,0,0,0,0))
    }
    
    finger_state = {}
    detected_fps = 0

    #video_capture.start()
    while True:
        if detected_fps > 14:
            detected_fps %= 15
        detected_fps += 1
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        #ret, frame = video_capture.read()
        frame = np.asanyarray(color_frame.get_data()).astype(np.uint8)

        center_X = int(frame.shape[1]/2)
        center_Y = int(frame.shape[0]/2)
        
        start = time.time()

        res = cv2.resize(frame, None, fx=FACTOR, fy=FACTOR)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        rects = detector(gray)    
        results = hands.process(rgb)
        
        cmd = np.zeros(6)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:       
                h, w, c = frame.shape   
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    cv2.circle(frame, (cx,cy), 3, (255, 0, 255), cv2.FILLED)
                gestion = gesture(handLms.landmark)
                hx, hy = gestion.hand_center()
    
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                finger_state[detected_fps%15] = gestion.finger_check()
                dep = gestion.depth_to_camera()

                #print(finger_state)

                if dep*frame.shape[1] > -50 or dep*frame.shape[1] <-30:
                    cmd += movs["look_forword"] * (dep*frame.shape[1] + 40) * 0.01

                distance = get_distance((hx*w, hy*h), (center_X, center_Y))
                if distance[0] > 100 or distance[0] < -10:
                    cmd += movs["look_left"] * distance[0] * 0.005
        
                if distance[1] > 100 or distance[1] < -100:
                    cmd += movs["look_up"] * distance[1] * 0.005
        
                cmd = np.clip(cmd, -0.3, 0.3)
        
                continue
        
        if check_up(finger_state):
                cmd = np.zeros(6)
                finger_state = {}
                #is_command = input('\n go home?(y/n):')
                #if is_command == 'y':
                print('up, go home')
                time.sleep(2)
                send_home(base_client_service)
        else:
            twist_command(base_client_service, list(cmd))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()

        # print(chr(27) + "[2J")  # Clear terminal
        # print("Center = {}\nVel Vec = {}\nFPS: {}\nFaces: {}".format((center_X, center_Y), cmd, 1/(end - start), len(rects)))

    #video_capture.stop()
    cv2.destroyAllWindows()

    send_home(base_client_service)

    print('Closing Session..')
    session_manager.CloseSession()
    router.SetActivationStatus(False)
    transport.disconnect()
    print('Done!')