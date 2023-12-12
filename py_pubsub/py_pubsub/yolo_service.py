import rclpy
from rclpy.node import Node
from yolo_strategy_interfaces.srv import YoloStrategy

import math
import cv2
import py_pubsub.darknet as darknet
import py_pubsub.pool_strategy as ps
import pyrealsense2 as rs

"""
神經網路檔案位置_檢測全部拼圖
"""
ALL_cfg_path ="/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/yolov4-obj.cfg"      #'./cfg/yolov4-obj.cfg'
ALL_weights_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/hiwin_C_WDA_v4.data'


"""
載入神經網路
"""
ALL_network, ALL_class_names, ALL_class_colors = darknet.load_network(
        ALL_cfg_path,
        ALL_data_path,
        ALL_weights_path,
        batch_size=1
)



"""
影像檢測
    輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
    輸出:(檢測後影像,檢測結果)
    註記:
"""
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
        #print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
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

def yaw_angle(vectorx, vectory):
    org_vector = [0,1]
    vectorlength = math.sqrt((vectorx**2)+(vectory**2))
    rad = math.acos((-1*vectory)/(vectorlength*1))
    theta = rad*180/math.pi
    if vectorx >= 0:
        return theta, rad
    elif vectorx < 0:
        return -theta, -rad
    
def pixel_mm_convert(pixel):
    actuallengh = 626
    pixellengh = 1920
    mm = actuallengh/pixellengh*pixel
    return mm

def realsense_intrinsics(x, y):
    width = 1920
    height = 1080
    fps = 30
    depth = 1
    
    # calibrated_intrinsics_f = [1362.38, 1360.45]
    # calibrated_intrinsics_pp = [938.315, 552.935]

    calibrated_intrinsics_f = [1366.72, 1364.65]
    calibrated_intrinsics_pp = [956.66, 567.60]

    # dis_coeffs = [0.0693826933, 0.445315521, 0.00291064076, -0.000845071017, -1.99098719]
    dis_coeffs = [0.1467682906, -0.43751602, 0.00113470242, -0.000273139029, 0.331039607]

    _intrinsics = rs.intrinsics()
    _intrinsics.width = width
    _intrinsics.height = height
    _intrinsics.ppx = calibrated_intrinsics_pp[0]
    _intrinsics.ppy = calibrated_intrinsics_pp[1]
    _intrinsics.fx = calibrated_intrinsics_f[0]
    _intrinsics.fy = calibrated_intrinsics_f[1]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = dis_coeffs

    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # ストリーミング開始

    # profile = pipeline.start(config)
    # depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
    # color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

    pixel = [x, y]
    ca_point = rs.rs2_deproject_pixel_to_point(_intrinsics, pixel, depth)
    # print('calibrated point:',ca_point)

    x_ = int(ca_point[0] * calibrated_intrinsics_f[0] + calibrated_intrinsics_pp[0])
    y_ = int(ca_point[1] * calibrated_intrinsics_f[1] + calibrated_intrinsics_pp[1])
    # print('calibrated intrinsics x:',x_)
    # print('calibrated intrinsics y:',y_)

    # pipeline.stop()
    return float(x_), float(y_)

def cam_angle_correction(camseex, camseey):
    width = 1920
    height = 1080
    if camseex <= width/2:
        midx = abs(width/2-camseex)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex+devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex+devx
            acty = camseey-devy
    
    else:
        midx = abs(camseex-width/2)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex-devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex-devx
            acty = camseey-devy

    return  actx, acty

def cam_angle_correction_2(camseex, camseey, camheight):
    width = 1920
    height = 1080
    if camseex <= width/2:
        midx = abs(width/2-camseex)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex+devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex+devx
            acty = camseey-devy
    
    else:
        midx = abs(camseex-width/2)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex-devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex-devx
            acty = camseey-devy

    return  actx, acty

class YOLOService(Node):
    def __init__(self):
        super().__init__('yolo_service')
        self.yolo_service = self.create_service(YoloStrategy, 'yolo_strategy', self.service_callback)
        self.n = 2201
        
        self.objectballx = []
        self.objectbally = []
        self.confidence = []
        self.intrin_objx = []
        self.intrin_objy = []
        self.corrected_objx = []
        self.corrected_objy = []
        self.lucky_flag = 0

    def service_callback(self, request, response):
        # detect image
        # clear array
        flat_list = []

        # request 1 for hitpoint position, yaw angle and route score
        if request.send_position == 1:
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('culculating hitpoint yaw and pitch angle \n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg')

            out2, ballinfo = detect_ALL(img)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3]-83)
                    #################3
                else:
                    cueindex = i
            self.confidence.append(flat_list[cueindex+1])
            self.objectballx.append(flat_list[cueindex+2])
            self.objectbally.append(flat_list[cueindex+3]-83)
            #####################

            # print("input objectball x:\n",self.objectballx)
            # print("input objectball y:\n",self.objectbally)
            n = len(self.objectballx)
            camheight = request.update_position[0]
            print(camheight)
            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
                # cam angle correction
                realx, realy = cam_angle_correction_2(intrinx, intriny, camheight)
                self.corrected_objx.append(realx)
                self.corrected_objy.append(realy)
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)
            print('real objectball x:\n',self.corrected_objx)
            print('real objectball y:\n',self.corrected_objy)


            ValidRoute, bestrouteindex, obstacle_flag = ps.main(self.corrected_objx[-1],self.corrected_objy[-1], 
                                                self.corrected_objx[0:n-1], self.corrected_objy[0:n-1],n-1)
            ########################
            # ps.main need to change upper and lower bound 
            

            print('All valid route:\n',ValidRoute)
            print('Best route index:',bestrouteindex)
            print('Best Route:\n',ValidRoute[bestrouteindex])
            print('Score of best route:\n',ValidRoute[bestrouteindex][0])

            hitpointx, hitpointy = ps.findhitpoint(self.corrected_objx[-1],self.corrected_objy[-1],
                                                ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            yaw, rad = yaw_angle(ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            
            # check if any obstacle behind hitpoint,  if so ajust pitch angle
            # obstacle_flag = ps.check_obstacle(self.corrected_objx[-1],self.corrected_objy[-1],
            #                                     ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1],
            #                                     self.corrected_objx, self.corrected_objy)
            
            # print('Any obstacle:', obstacle_flag)

            hitpointmmx = pixel_mm_convert(hitpointx)
            hitpointmmy = pixel_mm_convert(hitpointy)
            score = ValidRoute[bestrouteindex][0]
            print('degree:',yaw)
            response.current_position = [hitpointmmx, hitpointmmy, yaw, score]
            
        elif request.send_position == 2: # detect second photo 
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            
            self.get_logger().info('detecting second photo ! \n')
            img2 = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_second_ball.jpg')
            out2, ballinfo = detect_ALL(img2)
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_second_ball.jpg',out2)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # convert flat array to usable array 
            # usable array ==> in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    # self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3])
                    #################################
                    print(i)
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                # self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3])
                ############################
            else:
                pass

            # print("input objectball x:\n",self.objectballx)
            # print("input objectball y:\n",self.objectbally)
            n = len(self.objectballx)

            # DO CAMERA CALIBRATION HERE
            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
            
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)

            # return whole list, first half of list for x-axis and second half for y-axis
            wholelist = self.intrin_objx+self.intrin_objy
            print('1D ball position:\n', wholelist)
            response.current_position = wholelist

        elif request.send_position == 3: # send all ball in route
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('request for current ball location\n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg')

            out2, ballinfo = detect_ALL(img)
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_ball.jpg',out2)
            print('ball info:', ballinfo)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3]-83)
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3]-83)

            print("input objectball x:\n",self.objectballx)
            print("input objectball y:\n",self.objectbally)
            n = len(self.objectballx)

            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
                # cam angle correction
                realx, realy = cam_angle_correction(intrinx, intriny)
                self.corrected_objx.append(realx)
                self.corrected_objy.append(realy)
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)
            print('real objectball x:\n',self.corrected_objx)
            print('real objectball y:\n',self.corrected_objy)


            ValidRoute, bestrouteindex, ob_flag = ps.main(self.intrin_objx[-1],self.intrin_objy[-1], 
                                                self.intrin_objx[0:n-1], self.intrin_objy[0:n-1],n-1)
            
            score = ValidRoute[bestrouteindex][0]
            if score == -6000.0:
                self.lucky_flag = 1
            else:
                self.lucky_flag = 0
            
            # route() return this
            # score,cuefinalvector,cue,cuetoivector, objectballi, itok2vector, objectballk2 ,k2tok1vector, objectballk1, toholevector,n
            self.get_logger().info('sending best route ! \n')
            NBallInRoute = ValidRoute[bestrouteindex][-1] + 2 
            if ValidRoute[bestrouteindex][0] == -6000: #lcuky ball
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag
            elif NBallInRoute == 2:
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag
            elif NBallInRoute == 3:
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag
            else:
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][1]))
                # first index is cueball and the rest follow(s)
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag
    
        elif request.send_position == 4:
            re_position = request.update_position
            cuex = re_position[0]
            cuey = re_position[1]
            objx = []
            objy = []
            l = len(re_position)
            for i in range(2,len(re_position),2):
                objx.append(re_position[i])
                objy.append(re_position[i+1])
            m = int(l/2)-1
            # objx.append(cuex)
            # objy.append(cuey)
            
            
            # ValidRoute, bestrouteindex, ob_flag = ps.main(self.corrected_objx[-1],self.corrected_objy[-1], 
            #                                     self.corrected_objx[0:n-1], self.corrected_objy[0:n-1],n-1)

            # score = ValidRoute[bestrouteindex][0]
            # if score == -6000.0:
            #     self.lucky_flag = 1
            # else:
            #     self.lucky_flag = 0
            print('-----------------------------------------------')
            print('lucky flag value:',self.lucky_flag)
            print('cue x:',cuex)
            print('cue y:',cuey)
            print('obj x:',objx)
            print('obj y:',objy)
            print('-----------------------------------------------')
            if self.lucky_flag == 0:
                re_Route, re_bestindex, _ = ps.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps.findhitpoint(cuex, cuey,
                                                    re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                new_yaw, rad = yaw_angle(re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                new_hitpointmmx = pixel_mm_convert(new_hitpointx)
                new_hitpointmmy = pixel_mm_convert(new_hitpointy)
                score = re_Route[re_bestindex][0]

                response.current_position = [new_hitpointmmx, new_hitpointmmy, new_yaw, score]
                # response.current_position = [new_hitpointx, new_hitpointy, new_yaw, score]
            else:
                # # this is just for pausing, not lucky route
                # re_Route, re_bestindex, _ = ps.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps.hitpoint(cuex, cuey, objx[0]-cuex, objy[0]-cuey)
                new_yaw, rad = yaw_angle(objx[0]-cuex, objy[0]-cuey)
                new_hitpointmmx = pixel_mm_convert(new_hitpointx)
                new_hitpointmmy = pixel_mm_convert(new_hitpointy)
                score = -6000.0
                response.current_position = [new_hitpointmmx, new_hitpointmmy, new_yaw, score]

        else:
            self.get_logger().info('waiting for proper request...')
            response.current_position = [-3.0]
            
        return response
    
def main(args=None):
    rclpy.init(args=args)

    yolo_strategy = YOLOService()

    rclpy.spin(yolo_strategy)

    rclpy.shutdown()

if __name__=='__main__':
    main()