import pyrealsense2 as rs
import numpy as np
import cv2


# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
a = 0
# Start streaming
pipeline.start(config)


while True:

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # if not color_frame:
    #     continue

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    startoutline = (0,914)
    endoutline = (1920,914)
    table_outline = cv2.line(color_image,startoutline,endoutline,color=(0,0,0),thickness=1)
    startmidline = (960,0)
    endmidline = (960,914)
    middle_line = cv2.line(table_outline,startmidline,endmidline,color=(0,0,0),thickness=1)


    # Resize color image 
    resized_color_image = cv2.resize(middle_line, dsize=(1280,720), interpolation=cv2.INTER_AREA)    

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', resized_color_image)
    key=cv2.waitKey(1)
    if key&0xFF==ord('m'):
        a=a+1
        cv2.imwrite('/home/zack/docker_ws/ROS2_ws/src/IDPoolBall/Pictures/PoolBall_%d.jpg'% (a),middle_line)
        print("picture take")
    if key&0xFF==ord('q'):
        break



# Stop streaming
pipeline.stop()