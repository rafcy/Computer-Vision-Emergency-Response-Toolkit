""""
Submission For "Where is the Missing Person" Contest
Febuary 2020
Rafael Makrigiorgis - makrigiorgis.rafael@ucy.ac.cy
Panayiotis Kolios - kolios.panayiotis@ucy.ac.cy

"""
import time
import datetime
import cv2
import numpy
import threading
import imutils
import math
import os

###############################################
## Global for both Thermal/RGB VIDEO detection
###############################################
video_filename = 'DJI_0988.mov'
image_filename = 'train_MED_1030.jpg'
resize = True  # if you want to resize for better performance
save_results = False # False if you don't want to save detection/results in video/image format
im_width = 1080
im_height = 720
frameA = None
run_recv_thread = True

#########################################################################################
## variables for RGB detection ###########
#########################################################################################

# if you use image/video in RGB format you need to set the cfg/weights/class files for the NN detections
config = 'person.cfg'
cfg_width = 416
cfg_height = 416
weights = 'person.weights'
classes_file = 'person.names'

bboxes = []
network = None
classes = None
vid_out = None
video = None

###############################################
## variables for Thermal detection ##########
###############################################
iou_thresh = 15
cont_rs_thresh = 85
cont_allow_thresh = 35
gray = numpy.zeros((im_height, im_width, 1), numpy.uint8)
boxes = None
class box_in(object):
    def __init__(self, x=None, y=None, w=None, h=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class bount_boxes(object):
    def __init__(self,box=None, count_id=None, frame_id=None):
        self.box = box
        self.count_id=count_id
        self.frame_id = frame_id

###############################################

"""
Thermal Video Detection 
"""
#########################################################################################
## Functions for Thermal  Detection ######
#########################################################################################
def MidPoint(x1, y1, x2, y2):
    # Euclidean distance
    result = math.sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0))
    return result


def main_videoThermal():
    global video_filename
    global resize
    global im_width
    global im_height
    global  save_results
    global frameA
    global run_recv_thread
    global boxes
    global gray

    cap = cv2.VideoCapture(video_filename)
    ret, frame1 = cap.read()
    if resize:
        frame1 = cv2.resize(frame1, (im_width, im_height))

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = numpy.zeros_like(frame1)
    hsv[..., 1] = 255
    avg = None

    threading.Thread(target=recv_thread).start()
    key = None
    gray = numpy.zeros((im_height, im_width, 1), numpy.uint8)

    count = 0
    empty_count = 0
    frame_num = 0
    first_frame = False
    while run_recv_thread:
        if frameA is not None:
            frame2 = frameA.copy()
            if resize:
                frame = cv2.resize(frame2, (im_width, im_height))
            else:
                frame  = frame2
            if not first_frame:
                if save_results:
                    # initialize output video
                    Width = frame.shape[1]
                    Height = frame.shape[0]
                    videoname = video_filename.split('.')
                    path = "Detections/videos/"
                    try:
                        if not os.path.exists(path):
                            os.makedirs(path, 0o666)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)


                    vid_out = cv2.VideoWriter(path + 'det_' + videoname[0] + '.avi',
                                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 17.0, (Width, Height))
                    first_frame = True
            frame_num = frame_num + 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # next = gray
            cam_color_stat = (0.0, 255.0, 0.0)
            # if the average frame is None, initialize it
            if avg is None:
                print("[INFO] starting background model...")
                avg = gray.copy().astype("float")
                continue
            # find findContours
            cv2.accumulateWeighted(gray, avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            thresh = cv2.threshold(frameDelta, 5, 255,
                                   cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=4)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # loop over the contours
            none_boxes_flag = 0
            if boxes is None or boxes == []:
                boxes = []
                none_boxes_flag = 1
                # print("flag  = 1")

            # Getting the contours of the image
            # If the contours are a lot it means it has a lot of image movement so the code ignores it
            #   also it resets all the boxes saved if the movement was done 10 times.
            # If it has a number below a certain threshold the motion detection starts
            # First time initializing all the contours found on the image using bounding boxes
            # Then, on following frames, it checks the IOU score of each new contour box with all the previous boxes
            # If an IOU score is above a certain threshold it matches the box with the same ID and saves it into a list so it can track it
            # If the saved boxes for each box id are more than 5, it calculates the euclidean distance
            # if the euclidean distance is above a certatin thresh, it means it is moving and it draws the latest box and its trajectory
            # Also if a box(contour) is not detected for 20 frames, it resets the boxes of its ID so it will not match it with something else

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cw = 60  # custom width, the image from the video is cropped on the left/right sides
                         # cw = 0 for all other videos
                if (w - 120) * h > (im_width * im_height * 0.15):
                    continue
                if x <= cw or x + w >= im_width - cw:
                    continue


                if len(cnts) >= cont_rs_thresh:  # lots of changes means whole image is moving
                    if boxes is not None:
                        empty_count = empty_count + 1
                        if empty_count == 10:
                            boxes = []
                            empty_count = 0
                            # print("emptied")
                        # print(' shaky!!',empty_count)
                        cam_color_stat = (0.0, 0.0, 255.0)

                        break
                if len(cnts) <= cont_allow_thresh:
                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    box_temp = [round(x), round(y), round(x + w), round(y + h)]
                    cam_color_stat = (0.0, 255.0, 0.0)
                    count = 0
                    if none_boxes_flag == 1:
                        temp_box = []
                        temp_box.append(box_in(x, y, w, h))
                        boxes.append(bount_boxes(temp_box, count, frame_num))
                        none_boxes_flag = 0
                    # if boxes is not None:
                    else:
                        found = 0
                        for box in boxes:
                            latest_box = len(boxes[count].box) - 1
                            box_latest = box.box[latest_box]
                            box_temp2 = [round(box_latest.x), round(box_latest.y), round(box_latest.x + box_latest.w),
                                         round(box_latest.y + box_latest.h)]
                            if bb_iou(box_temp2, box_temp) >= iou_thresh: #IOU Threshold
                                bbox_temp = box.box
                                bbox_temp.append(box_in(x, y, w, h))
                                boxes[count].box = bbox_temp
                                boxes[count].frame_id = frame_num
                                box_count = 0
                                avg_eulc = 0

                                if len(boxes[count].box) > 5:
                                    lb = len(box.box) - 1
                                    # for num in range(lb-5,lb-1):
                                    avg_eulc = MidPoint(boxes[count].box[0].x, boxes[count].box[0].y,
                                                        boxes[count].box[lb].x, boxes[count].box[lb].y)
                                    if avg_eulc > 10:
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                        for bbox in box.box:
                                            cv2.circle(frame, (bbox.x + round(bbox.w / 2), bbox.y + round(bbox.h / 2)),
                                                       1, (0.0, 0.0, 255.0), 1)
                                found = 1
                                break
                            count = count + 1
                        if found == 0:
                            bbox_temp = []
                            bbox_temp.append(box_in(x, y, w, h))
                            idcount = len(boxes)
                            boxes.append(bount_boxes(bbox_temp, idcount, frame_num))

            cv2.putText(frame, 'Camera status', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color_stat, 1)

            if save_results:
                    vid_out.write(frame)
            prvs = next

            # showing the output image
            cv2.imshow('Detections', frame)

            count_b = 0
            for box in boxes:
                if box.frame_id + 20 <= frame_num:
                    bbox_temp = []
                    bbox_temp.append(box_in(0, 0, 0, 0))
                    boxes[count_b].box = bbox_temp
                count_b = count_b + 1

            key = cv2.waitKey(30) & 0xff

            if key == ord('q') or key == 27:
                run_recv_thread = False
                break

            count = count + 1
    cv2.destroyAllWindows()
    exit()
"""
RGB Video/Image Detection 
"""

#########################################################################################
## Functions for RGB  Detection #######
#########################################################################################

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,index):
    global classes

    label = str(classes[class_id])
    color = (0.0,255.0,255,0.0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    label = label + ' #' + str(index)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return

# Getting the IOU score for 2 boxes
def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou*100.0

#########################################################################################

"""
Processing RGB frames
Applying NN Detection of Person for each frame of video or image given
"""
def process_frame_RGB(frame):
    global network
    global frame_num
    global active_boxes
    global area
    global area_init

    # resizing image helps improve performance of the detection
    if resize:
        image = cv2.resize(frame, (im_width,im_height))
    else:
        image = frame

    if (image is not None):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # getting the detections
        blob = cv2.dnn.blobFromImage(image, scale, (cfg_width , cfg_height), (0, 0, 0), True, crop=False)
        network.setInput(blob)
        outs = network.forward(get_output_layers(network))

        class_ids = []
        confidences = []
        boxes = []
        # confidence and NMS threshold for the detections
        conf_threshold = 0.4
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        active_boxes = []
        index = 0
        # drawing the detections of each frame using bounding boxes
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            classid = class_ids[i]
            labelz = str(classes[classid])

            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                            round(y + h),index)
    return image

"""
Function for RGB Image Detection
"""

def main_imageRGGB():
    global frameA
    global run_recv_thread
    global network
    global classes
    global vid_out
    global video
    global frame_num
    global active_boxes
    global timer

    # Reading weights/config file given and initializing the Neural network
    network = cv2.dnn.readNet(weights, config)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
    print('Network initialized!')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    key = None

    frameA = cv2.imread('imgs/'+image_filename) # getting the image

    image = process_frame_RGB(frameA) #process image / detection
    if save_results:
        path ='Detections/images/'
        try:
            if not os.path.exists(path):
                os.makedirs(path, 0o666)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
        # saving the detection image
        cv2.imwrite(path+'det_' + image_filename, image)
    if resize:
        show_im = cv2.resize(image, (im_width,im_height))
    else:
        show_im = image
    # showing the output image
    cv2.imshow('Detection', show_im)
    cv2.waitKey(0)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        cv2.destroyAllWindows()
        exit()

"""
Function for RGB Video Detection
"""
def main_videoRGB():
    global frameA
    global run_recv_thread
    global network
    global classes
    global vid_out
    global video
    global frame_num
    global active_boxes
    global timer

    # Reading weights/config file given and initializing the Neural network
    network = cv2.dnn.readNet(weights, config)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
    print('Network initialized!')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    threading.Thread(target=recv_thread).start() # start the thread for getting the image
    key = None
    first_frame = 0
    while run_recv_thread:
        if frameA is None:  # if no img from the video is taken yet
            time.sleep(0.01)
        else:  # while getting image frames from the video
            image = process_frame_RGB(frameA)
            if resize:
                image = cv2.resize(image, (im_width, im_height))
            if not first_frame:
                if save_results:
                    # initialize output video
                    Width = image.shape[1]
                    Height = image.shape[0]
                    videoname = video_filename.split('.')
                    path = "Detections/videos/"
                    try:
                        if not os.path.exists(path):
                            os.makedirs(path, 0o666)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)


                    vid_out = cv2.VideoWriter(path + 'det_' + videoname[0] + '.avi',
                                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 17.0, (Width, Height))
                    first_frame = True
            # getting all the necessary processes on the image

            if save_results:
                vid_out.write(image)
            # showing the output image
            cv2.imshow('Detection', image)
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            run_recv_thread = False
            break

    # exits if  run_recv_thread = false
    print("Turning off camera.")
    video.release()
    if save_results:
        vid_out.release()
    print("Camera off.")
    print("Program ended.")
    cv2.destroyAllWindows()
    exit(0)

##############################################################################################
"""
thread for the video feed, helps the code to be running smoothly on every platform
"""
def recv_thread():

    global run_recv_thread
    global frameA
    global video
    global frame_num
    global timer
    print('start recv_thread()')

    video = cv2.VideoCapture(video_filename)
    # video = cv2.VideoCapture(0) # to get live camera feed
    print(run_recv_thread)
    while run_recv_thread:
        if video.isOpened():
            check, img = video.read()
            if check:
                frameA = img
                time.sleep(0.05) #0.05 for best results
            if check is not True:
                run_recv_thread = False
                break


##############################################################################################
"""
MAIN Function
"""
def main():
    # Set True ONLY the one you want to use
    video_rgb = False
    image_rgb =  False
    video_thermal = True

    print('Strating..')
    if video_rgb and not image_rgb  and not video_thermal :
        print('RGB video selected!')
        main_videoRGB()
    elif image_rgb and not video_rgb and not video_thermal:
        print('RGB image selected!')
        main_imageRGGB()
    elif video_thermal and not video_rgb and not  image_rgb:
        print('Thermal video selected!')
        main_videoThermal()
    else:
        print('You have to select only one method of detection!')


if __name__ == '__main__':
    main()