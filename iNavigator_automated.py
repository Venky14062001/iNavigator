# ALWAYS PRESS ESCAPE TO CLOSE CAMERA WINDOW
import cv2
import dlib
import math
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()  # to detect face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # to detect points on face
font = cv2.FONT_HERSHEY_SIMPLEX

def pointsForVerticalLine(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

def blinkDetremination(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    
    up_point = pointsForVerticalLine(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    down_point = pointsForVerticalLine(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    len_hLine = math.sqrt((left_point[0] - right_point[0]) ** 2 + (left_point[1] - right_point[1]) ** 2)
    len_vLine = math.sqrt((up_point[0] - down_point[0]) ** 2 + (up_point[1] - down_point[1]) ** 2)
    ratio = len_hLine / len_vLine
    return ratio

def gazeDetectionLR(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # SELECTING ONLY THE EYE FROM FACE
    h, w, _ = frame.shape
    mask = np.zeros((h, w), np.uint8)  # creating a black screen
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)  # filling the eye polygon with white color
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)  # need to see what it does

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    eye = frame[min_y: max_y, min_x: max_x]  # selecting the rectangular region with eye only
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # making gray scale
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)  # creating a threshold dont know exactly what it mean

    h, w = threshold_eye.shape
    left_Threshold = threshold_eye[0:int(h), 0:int(w / 2)]  # left part of threshold_eye window
    left_White = cv2.countNonZero(left_Threshold)  # zero mean black non zero mean white

    right_Threshold = threshold_eye[0:int(h), int(w / 2):int(w)]  # right part of threshold_eye window
    right_White = cv2.countNonZero(right_Threshold)
    if left_White == 0:
        gaze_ratio = 0.1
    elif right_White == 0:
        gaze_ratio = 3
    else:
        gaze_ratio = left_White / right_White
    return gaze_ratio




def eye_movement_detectLR(net_gaze_ratio):

    if net_gaze_ratio>=0 and net_gaze_ratio<=(Right_ear_list_avg+Center_ear_list_avg)/2:
        return "Right"
    elif net_gaze_ratio>=(Left_ear_list_avg+Center_ear_list_avg)/2 and net_gaze_ratio<=2.5:
        return "Left"
    else:
        return "Center"


modes_calibration = ['Left', 'Right', 'Center']

Left_ear_list = []
Right_ear_list = []
Center_ear_list = []

while len(modes_calibration) != 0:

    mode_input = input("Enter the mode you want to calibrate:{} ".format(modes_calibration))

    # Collect 100 frames to calibrate

    if mode_input in modes_calibration:

        if mode_input == "Left":
            for iterator in range(100):
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:  # faces is the 2d array containing the regio of face

                    landmarks = predictor(gray, face)  # the points on the face

                    # GAZE DETECTION

                    gaze_ratio_hor_left_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)
                    gaze_ratio_hor_right_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)

                    net_gaze_ratio_hor = (gaze_ratio_hor_left_eye + gaze_ratio_hor_right_eye) / 2

                cv2.imshow("frame", frame)

                Left_ear_list.append(net_gaze_ratio_hor)

        elif mode_input == "Right":
            for iterator in range(100):
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:  # faces is the 2d array containing the regio of face

                    landmarks = predictor(gray, face)  # the points on the face

                    # GAZE DETECTION

                    gaze_ratio_hor_right_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)
                    gaze_ratio_hor_left_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)

                    net_gaze_ratio_hor = (gaze_ratio_hor_left_eye + gaze_ratio_hor_right_eye) / 2

                cv2.imshow("frame", frame)

                Right_ear_list.append(net_gaze_ratio_hor)


        else:
            for iterator in range(100):
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:  # faces is the 2d array containing the regio of face

                    landmarks = predictor(gray, face)  # the points on the face

                    # GAZE DETECTION

                    gaze_ratio_hor_right_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)
                    gaze_ratio_hor_left_eye = gazeDetectionLR([42, 43, 44, 45, 46, 47], landmarks)

                    net_gaze_ratio_hor = (gaze_ratio_hor_left_eye + gaze_ratio_hor_right_eye) / 2

                cv2.imshow("frame", frame)

                Center_ear_list.append(net_gaze_ratio_hor)

        modes_calibration.remove(mode_input)

    else:
        print("Enter a valid mode!")

Left_ear_list_avg = sum(Left_ear_list) / len(Left_ear_list)
Right_ear_list_avg = sum(Right_ear_list) / len(Right_ear_list)
Center_ear_list_avg = sum(Center_ear_list) / len(Center_ear_list)



while (True):
    _, frame = cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:  # faces is the 2d array containing the region of face

        landmarks = predictor(gray, face)  # the points on the face

# BLINK DETECTION
        right_ratio = blinkDetremination([36,37,38,39,40,41], landmarks)
        left_ratio = blinkDetremination([42,43,44,45,46,47], landmarks)
        net_blink_ratio = (right_ratio + left_ratio)/2

        if left_ratio >=6:
            cv2.putText(frame, "left blink", (50,200), font, 1, (255,0,0) )
            pyautogui.click()
        if right_ratio >=6:
            cv2.putText(frame, "right blink", (50, 200), font, 1, (255, 0, 0))
            pyautogui.click(button='right')

# GAZE DETECTION

        gaze_ratio_left_eyeLR = gazeDetectionLR([42,43,44,45,46,47], landmarks)
        gaze_ratio_right_eyeLR = gazeDetectionLR([42,43,44,45,46,47], landmarks)
        

        net_gaze_ratioLR = (gaze_ratio_left_eyeLR + gaze_ratio_right_eyeLR)/2
        
        cv2.putText(frame, eye_movement_detectLR(net_gaze_ratioLR), (200,50), font, 2, (0,0,255), 3)

    

        if eye_movement_detectLR(net_gaze_ratioLR) == "Right":
            pyautogui.move(20, 0)
        if eye_movement_detectLR(net_gaze_ratioLR) == "Left":
            pyautogui.move(-20, 0)

        



        cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == 27:  # 27 ==escape key
        break

cap.release()
cv2.destroyAllWindows()