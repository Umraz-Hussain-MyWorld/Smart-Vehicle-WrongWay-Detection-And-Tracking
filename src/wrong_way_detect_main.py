# import neccesary libs
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from sort import * 

# load yoloV8 model for detection
detector = YOLO('../model/yolo8/yolov8n.pt')
number_plate_detetctor = YOLO('../model/yolo8/license_plate_detector.pt')

# Sort tracker
tracker  = Sort()

# if dont know the class id and name print it 
# print(f'{detector.names}')

# class indexes 
vehicle_index = [2, 3, 5, 7]


# vehicle list
vehicles_list = {2: 'car', 3: 'bike', 5: 'bus', 7: 'truck'}


# Create a directory for today's datetime inside ../data/output folder
output_dir = f"../data/output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
os.makedirs(output_dir, exist_ok=True)


# def function to get the coordinates of RIO's
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


write = True ## write video output

if write:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f'{output_dir}output_video.mp4'
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 1080))


# video path
path = '../data/input/test.mkv'

# read the video
cap = cv2.VideoCapture(path)

frame_no = 0
state = {}
count = {}
initial_value = {}
saved_images = {}
# num_plate = []

# loop through all the frames
while True:
    frame_no += 1
    ret, frame = cap.read()


    if ret : #and frame_no <= 1000:

        # resize the frames
        frame = cv2.resize(frame, (1920, 1080))

        # lets start detecting vehicles
        detections = detector(frame)[0]

        # list to append all vehicles 
        detections_ = []
        line_length_y = 20
        line_length_x = 20

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_index:
                detections_.append([x1, y1, x2, y2, score])

        # tracking 
        tracking_id = tracker.update(np.asarray(detections_))

        # # detect number plates
        # detected_plates = number_plate_detetctor(frame)[0]

        try:
            for bbox in tracking_id:
                x3, y3, x4, y4, id = bbox
                vx = int((x3 + x4) / 2)
                vy = int((y3 + y4) / 2)

                centroid = [int((x3 + x4) / 2), int((y3 + y4) / 2)] 

                # print(f'centroids : {centroid}')

                if id not in state.keys():    ## new entry to state dict
                    initial_value[id]=centroid[1]
                    state[id]=initial_value[id]
                    count[id]=0
                else:
                    state[id]=initial_value[id]-centroid[1]  ## update centroid's state
                text = f"ID {id}"

                '''
                check if state of centroid is non-negative . The negativity is determined by Y-axis of centroid
                if negative, the centroid will be monitored for 10 frames the centroid will turn Orange/Yellow
                in output window. If the state is negative for more than 10 frames , the centroid will be flagged
                as moving in wrong directiona nd bounding box turns Red 
                '''

                if state[id]>-20:
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                if state[id]<-20:
                    #print(objectID,state[objectID],count[objectID])
                    if count[id]>10:
                        count[id]+=1
                        cv2.putText(frame, f'Wrong Way: {text}', (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0, 255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0,255), -1)
                        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)

                        x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

                        cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0,0,255), 2)  #-- top-left
                        cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0,0,255), 2)

                        cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0,0,255), 2)  #-- bottom-left
                        cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0,0,255), 2)

                        cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0,0,255), 2)  #-- top-right
                        cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0,0,255), 2)

                        cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0,0,255), 2)  #-- bottom-right
                        cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0,0,255), 2)


                    if count[id]<=10:

                        if id not in saved_images:

                            cropped_img = frame[int(y3):int(y4), int(x3): int(x4), :]

                            # detetct and assign the number plate to vehicles
                            detected_plates = number_plate_detetctor(cropped_img)[0]

                            # if detected_plates is not None:
                            #     print(f'detected plates: {detected_plates}')
                            # else:
                            #     print('No plates detected')

                            for plate in detected_plates.boxes.data.tolist():
                                px1, py1, px2, py2, pScore, pId = plate

                                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)

                                cropped_plate = frame[int(py1):int(py2), int(px1): int(px2), :]

                                # save plate
                                cv2.imwrite(f"{output_dir}vehicle_{id}_wrong_way_image_plate-id_{pId}.jpg", cropped_plate)

                            # Crop and Save the image
                            # cropped_img = frame[int(y3):int(y4), int(x3): int(x4), :]
                            cv2.imwrite(f"{output_dir}vehicle_{id}_wrong_way_image.jpg", cropped_img)
                            saved_images[id] = True

                        cv2.putText(frame, f'Wrong Way: {text}', (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,165,255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,165,255), -1)
                        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0,165,255), 2)

                        x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

                        cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0,165,255), 2)  #-- top-left
                        cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0,165,255), 2)

                        cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0,165,255), 2)  #-- bottom-left
                        cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0,165,255), 2)

                        cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0,165,255), 2)  #-- top-right
                        cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0,165,255), 2)

                        cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0,165,255), 2)  #-- bottom-right
                        cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0,165,255), 2)
                        count[id]+=1
        except AttributeError:
            pass 

        # # save some frames for RIO's
        # frame_name = f'{frame_no}.jpg'
        # cv2.imwrite(f'{output_dir}/{frame_name}', frame)

        # Display the video 
        cv2.imshow('RGB', frame)

        if write:
            output_video.write(frame)

        # playback
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    else:
        print('Done Reading... or error reading video')
        break


# release the window
if write:
    output_video.release()
cap.release()
cv2.destroyAllWindows()
print('Done :) ')






































# old working final code

# # import neccesary libs
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import os
# from sort import * 

# # load yoloV8 model for detection
# detector = YOLO('../model/yolo8/yolov8n.pt')

# # Sort tracker
# tracker  = Sort()

# # if dont know the class id and name print it 
# # print(f'{detector.names}')

# # class indexes 
# vehicle_index = [2, 3, 5, 7]


# # vehicle list
# vehicles_list = {2: 'car', 3: 'bike', 5: 'bus', 7: 'truck'}

# vly1 = 349
# vly1_1 = 855
# vly2 = 855
# vly3 = 349
# offSet = 75



# # Create a directory for today's datetime inside ../data/output folder
# output_dir = f"../data/output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
# os.makedirs(output_dir, exist_ok=True)


# # def function to get the coordinates of RIO's
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)
        

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)


# write=False ## write video output

# if write:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video_path = '../data/output/output_video.mp4'
#     output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 1080))


# # video path
# path = '../data/input/test.mkv'

# # read the video
# cap = cv2.VideoCapture(path)


# frame_no = 0
# vUpl2 = {}
# vUpl3 = {}
# vRight1 = {}
# vDownWrongL2 = {}
# vDownWrongL3 = {}
# state = {}
# count = {}
# initial_value = {}



# # loop through all the frames
# while True:
#     frame_no += 1
#     ret, frame = cap.read()


#     if ret and frame_no <= 1000:

#         # resize the frames
#         frame = cv2.resize(frame, (1920, 1080))

#         # lets start detecting vehicles
#         detections = detector(frame)[0]

#         # list to append all vehicles 
#         detections_ = []
#         line_length_y = 20
#         line_length_x = 20

#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicle_index:
#                 detections_.append([x1, y1, x2, y2, score])
#                 # cv2.putText(frame, f'{vehicles_list[int(class_id)]}', (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
#                 # x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#                 # cv2.line(frame, (x1, y1), (x1, y1 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 # cv2.line(frame, (x1, y1), (x1 + line_length_x, y1), (0, 255, 0), 2)

#                 # cv2.line(frame, (x1, y2), (x1, y2 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 # cv2.line(frame, (x1, y2), (x1 + line_length_x, y2), (0, 255, 0), 2)

#                 # cv2.line(frame, (x2, y1), (x2 - line_length_x, y1), (0, 255, 0), 2)  #-- top-right
#                 # cv2.line(frame, (x2, y1), (x2, y1 + line_length_y), (0, 255, 0), 2)

#                 # cv2.line(frame, (x2, y2), (x2, y2 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 # cv2.line(frame, (x2, y2), (x2 - line_length_x, y2), (0, 255, 0), 2)

#         # tracking 
#         tracking_id = tracker.update(np.asarray(detections_))

#         try:
#             for bbox in tracking_id:
#                 x3, y3, x4, y4, id = bbox
#                 vx = int((x3 + x4) / 2)
#                 vy = int((y3 + y4) / 2)

#                 centroid = [int((x3 + x4) / 2), int((y3 + y4) / 2)] 

#                 # print(f'centroids : {centroid}')

#                 if id not in state.keys():    ## new entry to state dict
#                     initial_value[id]=centroid[1]
#                     state[id]=initial_value[id]
#                     count[id]=0
#                 else:
#                     state[id]=initial_value[id]-centroid[1]  ## update centroid's state
#                 text = f"ID {id}"

#                 '''
#                 check if state of centroid is non-negative . The negativity is determined by Y-axis of centroid
#                 if negative, the centroid will be monitored for 10 frames the centroid will turn Orange/Yellow
#                 in output window. If the state is negative for more than 10 frames , the centroid will be flagged
#                 as moving in wrong directiona nd bounding box turns Red 
#                 '''

#                 if state[id]>-20:
#                     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
#                     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

#                 if state[id]<-20:
#                     #print(objectID,state[objectID],count[objectID])
#                     if count[id]>10:
#                         count[id]+=1
#                         cv2.putText(frame, f'Wrong Way: {text}', (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0, 255), 2)
#                         cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0,255), -1)
#                         # cv2.rectangle(frame, (x3, y3), (x4, y4), (0,0,255), 2)

#                         # cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                         # cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                         # cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                         # cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                         # cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                         # cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                         # cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                         # cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)


#                     if count[id]<=10:
#                         cv2.putText(frame, f'Wrong Way: {text}', (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,165,255), 2)
#                         cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,165,255), -1)
#                         # cv2.rectangle(frame, (x3, y3), (x4, y4), (0,165,255), 2)

#                         # cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                         # cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                         # cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                         # cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                         # cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                         # cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                         # cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                         # cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)


#                         count[id]+=1


#                 # # vehicles passing from line 2 to 2.1
#                 # if 855 < (vy + offSet) and 855 > (vy - offSet):
#                 #     vUpl2[id] = vy

#                 # if id in vUpl2:
#                 #     if 923 < (vy + offSet) and 923 > (vy - offSet):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 #         cv2.putText(frame, f'Wrong Way', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)





#                 # # vehicles passing from line 2.1 to 2
#                 # if 923 < (vy + offSet) and 923 > (vy - offSet):
#                 #     vUpl2[id] = vy

#                 # if id in vUpl2:
#                 #     if 855 < (vy + offSet) and 855 > (vy - offSet):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#                 #         cv2.putText(frame, f'Going straight', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)


                
#                 # if 349 < (vy + 30) and 349 > (vy - 30):
#                 #     vUpl3[id] = vy

#                 # if id in vUpl3:
#                 #     if 309 < (vy + 50) and 309 > (vy - 50):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
#                 #         cv2.putText(frame, f'Going Straight', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
#                 #     # else:
#                 #     #     cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
#                 #     #     cv2.putText(frame, f'None', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)


#                 # if 309 < (vy + 6) and 309 > (vy - 6):
#                 # #     vDownWrongL3[id] = vy

#                 # # if id in vDownWrongL3:
#                 #     if 349 < (vy + 70) and 349 > (vy - 70):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 #         cv2.putText(frame, f'Wrong Way', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)




#                 # if 855 < (vy + 6) and 855 > (vy - 6):
#                 #     vDownWrong[id] = vy

#                 # if id in vDownWrong:
#                 #     if 855 < (vy + 3) and 855 > (vy - 3):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 #         cv2.putText(frame, f'Wrong Way', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



#                 # if (319 < (vy + 6) and 781 > (vy - 6)) or (1304 < (vx + 6) and 1768 > (vx - 6)):
#                 #     vRight1[id] = vy

#                 # if id in VRight1:
#                 #     if (319 < (vy + 20) and 781 > (vy - 20)) or (1304 < (vx + 20) and 1768 > (vx - 20)):
#                 #         x3, y3, x4, y4 = map(int, [x3, y3, x4, y4])

#                 #         cv2.line(frame, (x3, y3), (x3, y3 + line_length_y), (0, 255, 0), 2)  #-- top-left
#                 #         cv2.line(frame, (x3, y3), (x3 + line_length_x, y3), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x3, y4), (x3, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-left
#                 #         cv2.line(frame, (x3, y4), (x3 + line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y3), (x4 - line_length_x, y3), (0, 255, 0), 2)  #-- top-right
#                 #         cv2.line(frame, (x4, y3), (x4, y3 + line_length_y), (0, 255, 0), 2)

#                 #         cv2.line(frame, (x4, y4), (x4, y4 - line_length_y), (0, 255, 0), 2)  #-- bottom-right
#                 #         cv2.line(frame, (x4, y4), (x4 - line_length_x, y4), (0, 255, 0), 2)

#                 #         cv2.circle(frame, (vx, vy), 4, (0, 0, 255), -1)
#                 #         cv2.putText(frame, f'ID - {id}', (vx, vy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#                 #         cv2.putText(frame, f'Going Left', (vx, vy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
#         except AttributeError:
#             pass 

#         # # save some frames for RIO's
#         # frame_name = f'{frame_no}.jpg'
#         # cv2.imwrite(f'{output_dir}/{frame_name}', frame)

#         # # put lines - line-1
#         # cv2.line(frame, (1403, 317), (1855, 743), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-1 ^', (1859, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

#         # # put lines - line-1.1
#         # cv2.line(frame, (1304, 319), (1768, 781), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-1.1 ^', (1784, 771), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 250), 2)

#         # # line-2
#         # cv2.line(frame, (1714, 855), (449, 855), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-2 >', (423, 839), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#         # # line-2.1
#         # cv2.line(frame, (1710, 923), (423, 923), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-2.1 >', (400, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 250), 2)

#         # # line-3
#         # cv2.line(frame, (1222, 349), (645, 349), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-3 >', (640, 341), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#         # # line-3.1
#         # cv2.line(frame, (675, 309), (1061, 309), (255, 255, 0), 2)
#         # cv2.putText(frame, f'line-3.1 >', (659, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 250), 2)


#         # Display the video 
#         cv2.imshow('RGB', frame)

#         # playback
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord("q"):
#             break


# # release the window
# cap.release()
# cv2.destroyAllWindows()