import cv2
import numpy as np

import pyrealsense2 as rs

class Gen3Lite_video:
    def __init__(self):
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
