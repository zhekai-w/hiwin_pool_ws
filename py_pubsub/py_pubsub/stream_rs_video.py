## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import darknet 



"""
神經網路檔案位置_檢測全部拼圖
"""
ALL_cfg_path = './cfg/yolov4-obj.cfg'
ALL_weights_path = './cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = './cfg/hiwin_C_WDA_v4.data'


"""
載入神經網路
"""
ALL_network, ALL_class_names, ALL_class_colors = darknet.load_network(
        ALL_cfg_path,
        ALL_data_path,
        ALL_weights_path,
        batch_size=1
)

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



"""
座標轉換
    輸入:(YOLO座標,原圖寬度,原圖高度)
    輸出:(框的左上座標,框的右下座標)
    註記:
"""
def bbox2points(bbox,W,H):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """ 
    width = darknet.network_width(ALL_network)      # YOLO壓縮圖片大小(寬)
    height = darknet.network_height(ALL_network)    # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
    x = x*W/width
    y = y*H/height
    w = w*W/width
    h = h*H/height
    # 輸出框座標_YOLO格式
    # print("     (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(x, y, w, h))
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    
    return xmin, ymin, xmax, ymax



"""
原圖繪製檢測框線
    輸入:(檢測結果,原圖位置,框線顏色集)
    輸出:(影像結果)
    註記:
"""
def draw_boxes(detections, image, colors):
    ball_imformation = [[-999 for i in range(4)] for j in range(20)]
    i = 0

    H,W,_ = image.shape                      # 獲得原圖長寬

    # cv2.line(image,(640,0),(640,720),(0,0,255),5)

    for label, confidence, bbox in detections:
        xmin, ymin, xmax, ymax = bbox2points(bbox,W,H)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        # 輸出框座標_加工格式座標(左上點座標,右上點座標)
        print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
        mx = float(xmax + xmin)/2
        my = float(ymax + ymin)/2

        # cv2.circle(image, (int(mx),int(my)), 33, (0,0,255), 3)
        if label == 'C':
            ball_imformation[i] = [0.0, float(confidence), mx, my]
        elif label == 'M':
            ball_imformation[i] = [1.0, float(confidence), mx, my]
        i+=1
        

    return image, ball_imformation


def detect_ALL(img,thresh=0.8):
    out,detections = image_detection(img,ALL_network, ALL_class_names, ALL_class_colors,thresh)
    out2, ball_imformation= draw_boxes(detections, img, ALL_class_colors)

    # cv2.imshow('out2', out2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return out2, ball_imformation


if __name__ == "__main__": 

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
            color_img = np.asanyarray(color_frame.get_data())
            detected_img = detect_ALL(color_img)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', detected_img)
            key=cv2.waitKey(1)
            # if key&0xFF==ord('m'):
                # print("picture take")
                # cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/vefity_pics.jpg',table_outline_bot)
            if key&0xFF==ord('q'):
                cv2.destroyAllWindows()
                break

    # Stop streaming
    pipeline.stop()
