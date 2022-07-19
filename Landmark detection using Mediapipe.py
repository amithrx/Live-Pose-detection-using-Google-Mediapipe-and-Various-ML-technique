#hard coded algorithm for pose detection
from ctypes import sizeof
from pickle import NONE
from turtle import distance
import cv2
from cv2 import COLOR_BGR2RGB
import mediapipe as mp
import numpy as np   
import time 
from datetime import datetime
import csv
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#VIDEO FEED
flag = False
counter = -1
# stage=NONE

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # if angle >180.0:
    #     angle = 360-angle
        
    return angle 

def calculate_distance(a,b):
  a = np.array(a)
  b = np.array(b)
  distance = np.sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1]))
  return distance

header = ['nose_x_1','l_shouldr_x_1','r_shoulder_x_1','l_elbow_x_1','r_elbow_x_1','l_wrist_x_1','r_wrist_x_1',
          'l_hip_x_1','r_hip_x_1','l_knee_x_1','r_knee_x_1','l_ankle_x_1','r_ankle_x_1','nose_y_1','l_shouldr_y_1',
          'r_shoulder_y_1','l_elbow_y_1','r_elbow_y_1','l_wrist_y_1','r_wrist_y_1','l_hip_y_1','r_hip_y_1','l_knee_y_1',
          'r_knee_y_1','l_ankle_y_1','r_ankle_y_1','nose_z_1','l_shouldr_z_1','r_shoulder_z_1','l_elbow_z_1','r_elbow_z_1',
          'l_wrist_z_1','r_wrist_z_1','l_hip_z_1','r_hip_z_1','l_knee_z_1','r_knee_z_1','l_ankle_z_1','r_ankle_z_1','nose_x_2',
          'l_shouldr_x_2','r_shoulder_x_2','l_elbow_x_2','r_elbow_x_2','l_wrist_x_2','r_wrist_x_2','l_hip_x_2','r_hip_x_2',
          'l_knee_x_2','r_knee_x_2','l_ankle_x_2','r_ankle_x_2','nose_y_2','l_shouldr_y_2','r_shoulder_y_2','l_elbow_y_2',
          'r_elbow_y_2','l_wrist_y_2','r_wrist_y_2','l_hip_y_2','r_hip_y_2','l_knee_y_2','r_knee_y_2','l_ankle_y_2',
          'r_ankle_y_2','nose_z_2','l_shouldr_z_2','r_shoulder_z_2','l_elbow_z_2','r_elbow_z_2','l_wrist_z_2',
          'r_wrist_z_2','l_hip_z_2','r_hip_z_2','l_knee_z_2','r_knee_z_2','l_ankle_z_2','r_ankle_z_2','nose_x_3',
          'l_shouldr_x_3','r_shoulder_x_3','l_elbow_x_3','r_elbow_x_3','l_wrist_x_3','r_wrist_x_3','l_hip_x_3','r_hip_x_3',
          'l_knee_x_3','r_knee_x_3','l_ankle_x_3','r_ankle_x_3','nose_y_3','l_shouldr_y_3','r_shoulder_y_3','l_elbow_y_3',
          'r_elbow_y_3','l_wrist_y_3','r_wrist_y_3','l_hip_y_3','r_hip_y_3','l_knee_y_3','r_knee_y_3','l_ankle_y_3',
          'r_ankle_y_3','nose_z_3','l_shouldr_z_3','r_shoulder_z_3','l_elbow_z_3','r_elbow_z_3','l_wrist_z_3',
          'r_wrist_z_3','l_hip_z_3','r_hip_z_3','l_knee_z_3','r_knee_z_3','l_ankle_z_3','r_ankle_z_3','output']
data = []
data_inserted = []
output = []
temporay_list = []
check = 0
prev = -1
def removNestings(l):
    for i in l:
        if type(i) == list:
            removNestings(i)
        else:
            output.append(i)

cnt=0
cnt_list=0
dnt=0
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            # Normal landmarks
            landmarks = results.pose_landmarks.landmark 
            # World landmarks
            # landmarks = results.pose_world_landmarks.landmark
            
            # Get coordinates
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # Calculate angle
            angle_bend = calculate_angle(l_shoulder, l_hip, l_knee)
            distance_run = calculate_distance(l_ankle, l_hip)
            distance_squat = abs(landmarks[mp_pose.PoseLandmark.NOSE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            distance_squat1 = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            distance_bend = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            distance_bend1 = abs(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            angle_run = calculate_angle(l_ankle,l_knee,l_hip)

            # finding datasets
            print("x ",landmarks[mp_pose.PoseLandmark.NOSE.value].x)
            print("y ",landmarks[mp_pose.PoseLandmark.NOSE.value].y)
            print("z ",landmarks[mp_pose.PoseLandmark.NOSE.value].z)

            another_list = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                            landmarks[mp_pose.PoseLandmark.NOSE.value].z,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,2]
                           
                            
                            
                                          
            begin = datetime.now()
            begin = begin.second
            if check==0:
              prev=begin
              check=1
            # print("begin",begin)
            current = datetime.now().second
            # print("current",datetime.now().second)
            curr = datetime.now()
            # print("current_micro",curr.microsecond)  
            
            temporay_list.append(another_list)
            # print(temporay_list)
            print(cnt)
            if current==prev and current==begin:
              cnt+=1
            else:
              dnt=cnt/2
              cnt=0
              check=0
            if cnt==0:
              # print("hii")
              data.append(temporay_list[0])
              data.append(temporay_list[dnt])
              data.append[temporay_list[temporay_list.size()-1]]
              temporay_list=[]
            # print(data)
            if len(data)==3:
              removNestings(data)
              idx_list=[39,79]
              res = [ele for idx, ele in enumerate(output) if idx not in idx_list]
              data_inserted.append(res)
              cnt_list+=1
              output=[]
              data.pop(0)

            if flag==False:
              current_left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
              flag=True
            
            
            # Hard codings
            # counter=-1
            # if distance_bend>=0.60 and angle_bend>=165 and angle_bend<=182 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= current_left_shoulder+0.02 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y >= current_left_shoulder-0.02 and angle_run<=185 and angle_run>=175:
            #   counter=2
            # elif angle_bend>=165 and angle_bend<=182 and (angle_run>=179 or angle_run<=175) and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= current_left_shoulder+0.04 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y >= current_left_shoulder-0.05 and distance_squat1>=0.28 and distance_squat>=0.4 and distance_bend>=0.56 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y<=0.53:
            #   counter=4
            # elif landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < current_left_shoulder-0.05 and angle_run<=182 and angle_run>=176:
            #   counter=5
            # if angle_bend>185 and distance_squat1>0.33 and distance_bend<=0.74:
            #   counter=1
            # if angle_bend<156 and distance_squat>0.25 and distance_squat1>0.32 and angle_run>=165 and angle_run<=200 and distance_bend1<=0.72:
            #   counter=0

            
            # if angle_run>198 and distance_squat<0.3 and distance_squat1<0.37:
            #   counter=3

            # x = counter
            # my_list = []
            # while x == 5:
            #   print ("enter")
            #   l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            #   l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            #   l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            #   l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            #   nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            #   l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # # Calculate angle
            #   angle_bend = calculate_angle(l_shoulder, l_hip, l_knee)
            #   distance_run = calculate_distance(l_ankle, l_hip)
            #   distance_squat = abs(landmarks[mp_pose.PoseLandmark.NOSE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            #   distance_squat1 = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            #   distance_bend = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            #   distance_bend1 = abs(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y-landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            #   angle_run = calculate_angle(l_ankle,l_knee,l_hip)

            #   index = -1
            #   if distance_bend>=0.60 and angle_bend>=165 and angle_bend<=182 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= current_left_shoulder+0.02 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y >= current_left_shoulder-0.02 and angle_run<=185 and angle_run>=175:
            #     my_list.append("straight")
            #   # elif angle_bend>=165 and angle_bend<=182 and (angle_run>=179 or angle_run<=175) and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= current_left_shoulder+0.04 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y >= current_left_shoulder-0.05 and distance_squat1>=0.28 and distance_squat>=0.4 and distance_bend>=0.56 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y<=0.53:
            #   #   my_list.append("run")
            #   # elif landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < current_left_shoulder-0.05 and angle_run<=182 and angle_run>=176:
            #   #   my_list.append("jump")
            #     # index=0
            #   for amit in my_list:
            #     print (amit)
            #   stage="jump"
            #   # if my_list.index("jump")<sizeof(my_list) and my_list.index("jump")>=0:
            #   #   index = my_list.index("jump")
            #   if index==-1 and sizeof(my_list)>100:
            #     x=2
                
            #   else:
            #     cv2.putText(image, stage, 
            #         (60,60), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            #     my_list.clear()
            # counter=x


            # while counter==5:
            #   if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < current_left_shoulder-0.03 and angle_run<=182 and angle_run>=176:
            # dtage="down"
            # if angle_run>=200 and distance_squat>0.35 and dtage=="down":
            #   dtage="up"
            # if angle_run>=200 and distance_squat>0.35 and dtage=="up":
            #   dtage="down"
            #   counter=4
            

        except:
            pass
      
        # Setup status box
        cv2.rectangle(image, (0,0), (425,73), (245,117,16), -1)
        
        stage = "NONE"
        if counter==4:
          stage="run"
        if counter==5:
          stage="jump"
        if counter==1:
          stage="left_bend"
        if counter==0:
          stage="right_bend"
        if counter==3:
          stage="squat"
        if counter==2:
          stage="straight"
        if counter==-1:
          stage="NONE"
       
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
    
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        
        # Press q for quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # with open('magnimus.csv' , 'w' , newline='') as f:
    #           writer = csv.writer(f)
    #           writer.writerow(header)
    #           # print(data)
    #           writer.writerows(data_inserted)


