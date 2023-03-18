import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import threading
import time
import pyautogui
import numpy as np
import mouse
import pandas as pd
import math
webcam = 0
exit_status = -1
go_right = 0
go_left = 0
go_up = 0
go_down = 0
ges_ctrl_track = 0
gesture_on = None
misle_coldwn = 0
flare_coldwn = 0
fire_on = 0
booster_coldwn =0


# Racing

def r_side_steer():
    global right_steer_thread
    global right_hand_op
    pyautogui.keyDown("Left")
    pyautogui.keyUp("Right")
    print("Right")
    right_steer_thread = threading.Thread(target=r_side_steer)

def l_side_steer():
    global left_steer_thread
    global left_hand_op
    pyautogui.keyDown("Right")
    pyautogui.keyUp("Left")
    print("Left")
    left_steer_thread = threading.Thread(target= l_side_steer)

def s_side_steer():
    global strght_steer_thread
    pyautogui.keyUp("Left")
    pyautogui.keyUp("Right")
    print("Straight")
    strght_steer_thread = threading.Thread(target= s_side_steer)


def nitro_coldwn_func():
    global nitro_coldwn
    global nitro_coldwn_thread
    time.sleep(2)
    nitro_coldwn = 0
    print("nitros reloaded")
    nitro_coldwn_thread = threading.Thread(target=nitro_coldwn_func)


left_steer_on = 0
right_steer_on = 0
strght_steer_on = 1
nitro_coldwn = 0
brake_on = 0

right_steer_thread = threading.Thread(target=r_side_steer)
left_steer_thread = threading.Thread(target= l_side_steer)
strght_steer_thread = threading.Thread(target= s_side_steer)
nitro_coldwn_thread = threading.Thread(target=nitro_coldwn_func)















def exit_stat_change():
    global exit_status
    print("change")
    exit_status = 0
def ges_enable():
    global gesture_on
    global gesture_on_thread
    time.sleep(3)
    gesture_on = 1
    print("Gesture Control On")
    gesture_on_thread = threading.Thread(target=ges_enable)

def misle_cooldown_func():
    global misle_coldwn
    global misle_coldwn_thread
    time.sleep(2)
    misle_coldwn = 0
    print("Missle reloaded")
    misle_coldwn_thread = threading.Thread(target=misle_cooldown_func)

def flare_coldwn_func():
    global flare_coldwn
    global flare_coldwn_thread
    time.sleep(2)
    flare_coldwn = 0
    print("Flares reloaded")
    flare_coldwn_thread = threading.Thread(target=flare_coldwn_func)

def booster_coldwn_func():
    global booster_coldwn
    global booster_coldwn_thread
    time.sleep(2)
    booster_coldwn = 0
    print("Boosters cooling down")
    booster_coldwn_thread = threading.Thread(target=booster_coldwn_func)


def right_hand_dual():
    global right_dual_thread
    global right_hand_op
    pyautogui.keyDown("Right")
    right_dual_thread = threading.Thread(target=right_hand_dual)


def left_hand_dual():
    global left_dual_thread
    global left_hand_op
    pyautogui.keyDown("Left")
    left_dual_thread = threading.Thread(target=left_hand_dual)


right_dual_thread = threading.Thread(target=right_hand_dual)
left_dual_thread = threading.Thread(target=left_hand_dual)




gesture_on_thread = threading.Thread(target=ges_enable)
misle_coldwn_thread = threading.Thread(target=misle_cooldown_func)
flare_coldwn_thread = threading.Thread(target=flare_coldwn_func)
booster_coldwn_thread = threading.Thread(target=booster_coldwn_func)
right_dual_thread = threading.Thread(target=right_hand_dual)
left_dual_thread = threading.Thread(target= left_hand_dual)

def activate(mode,mode_ctrl_state):
    global webcam
    cap = cv2.VideoCapture(webcam)
    cam_w, cam_h = 640, 480
    cap.set(3, cam_w)
    cap.set(4, cam_h)
    frameR = 100
    global go_right, go_left, go_up, go_down, ges_ctrl_track, gesture_on, misle_coldwn, flare_coldwn,fire_on, booster_coldwn
    global gesture_on_thread, misle_coldwn_thread, flare_coldwn_thread, booster_coldwn_thread
    global exit_status
    global right_dual_thread,left_dual_thread
    charac_pos = [0, 1, 0]
    index_pos = 1
    fixedx = None
    fixedy = None
    rec = None

    global left_steer_thread,right_steer_thread,strght_steer_thread,nitro_coldwn_thread
    global left_steer_on, right_steer_on, strght_steer_on, nitro_coldwn, brake_on

    ######## Csv Control Files ################
    df_flight = pd.read_csv("flight_ctrl.csv")
    df_sub = pd.read_csv("sub_ctrl.csv")
    go_right = 0
    go_left = 0
    go_up = 0
    go_down = 0
    ges_ctrl_track = 0
    gesture_on = None
    misle_coldwn = 0
    flare_coldwn = 0
    fire_on = 0
    booster_coldwn = 0
    print("load")
    left_hand_op = 0
    right_hand_op = 0



    fpsReader = cvzone.FPS()
    face_detect = FaceMeshDetector(maxFaces=1)
    if mode ==2:
        hand_detect = HandDetector(maxHands=2, detectionCon=0.5)
    else:
        hand_detect = HandDetector(maxHands=2, detectionCon=0.75)
  #  hand_detect = HandDetector(maxHands=2, detectionCon=0.5)
    pose_detect = PoseDetector(detectionCon= 0.75, trackCon=0.7)
    exit_status = -1
    while True:
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)

            #Aircraft
            if mode== 3:
                img, faces = face_detect.findFaceMesh(img, draw=False)

                if faces:
                    ctrl_x, ctrl_y = faces[0][168][0], faces[0][168][1]
                    left_x, left_y = faces[0][57][0] + 10, faces[0][57][1]
                    right_x, right_y = faces[0][287][0] - 10, faces[0][287][1]

                    cv2.circle(img, (ctrl_x, ctrl_y), 5, (255, 255, 0), 2)
                    cv2.circle(img, (left_x, left_y), 5, (255, 255, 0), 2)
                    cv2.circle(img, (right_x, right_y), 5, (255, 255, 0), 2)
                    cv2.line(img, (ctrl_x, 0), (ctrl_x, 1000), (255, 255, 255), 2)
                    cv2.line(img, (0, ctrl_y), (1000, ctrl_y), (255, 255, 255), 2)

                    hands, img = hand_detect.findHands(img, draw=True, flipType=False)
                    if len(hands) == 2:
                        if hands[0]['type'] == "Left":
                            hands_l, hands_r = hands[0], hands[1]
                        else:
                            hands_l, hands_r = hands[1], hands[0]

                        lmlist_l, lmlist_r = hands_l['lmList'], hands_r['lmList']

                        r_idx_x, r_idx_y = lmlist_r[6][0], lmlist_r[6][1]
                        l_idx_x, l_idx_y = lmlist_l[6][0], lmlist_l[6][1]
                        r_thmb_x, r_thmb_y = lmlist_r[4][0], lmlist_r[4][1]
                        l_thmb_x, l_thmb_y = lmlist_l[4][0], lmlist_l[4][1]

                        cv2.circle(img, (r_idx_x, r_idx_y), 5, (0, 255, 255), 2)
                        cv2.circle(img, (l_idx_x, l_idx_y), 5, (0, 255, 255), 2)
                        cv2.circle(img, (r_thmb_x, r_thmb_y), 5, (0, 255, 255), 2)
                        cv2.circle(img, (l_thmb_x, l_thmb_y), 5, (0, 255, 255), 2)

                        l_fing_upno = hand_detect.fingersUp(hands_l)
                        r_fing_upno = hand_detect.fingersUp(hands_r)

                        if l_fing_upno[0] == 1 and l_fing_upno[4] == 1 and r_fing_upno[0] == 1 and r_fing_upno[4] == 1 and \
                                sum(l_fing_upno) == 2 and sum(r_fing_upno) == 2 and ges_ctrl_track == 0:
                            updwn_line_control = ctrl_y
                            ges_ctrl_track = 1
                            print("Gesture Enabled")
                            gesture_on_thread.start()

                if gesture_on == 1:
                    cv2.line(img, (0, updwn_line_control), (1000, updwn_line_control), (255, 0, 255), 2)

                    # Aircraft Moving Up
                    if abs(updwn_line_control - ctrl_y) > 20 and updwn_line_control > ctrl_y and go_up == 0:
                        pyautogui.keyDown(df_flight["up"][mode_ctrl_state])
                        go_up = 1
                        print("Going Up")
                    elif abs(updwn_line_control - ctrl_y) < 20 and updwn_line_control > ctrl_y and go_up == 1:
                        pyautogui.keyUp(df_flight["up"][mode_ctrl_state])
                        go_up = 0

                    # Aircraft Moving down
                    if abs(updwn_line_control - ctrl_y) > 35 and updwn_line_control < ctrl_y and go_down == 0:
                        pyautogui.keyDown(df_flight["down"][mode_ctrl_state])
                        go_down = 1
                        print("Going Down")
                    elif abs(updwn_line_control - ctrl_y) < 35 and updwn_line_control < ctrl_y and go_down == 1:
                        pyautogui.keyUp(df_flight["down"][mode_ctrl_state])
                        go_down = 0

                    # Moving the Aircraft to the Right
                    if right_x < ctrl_x and go_right == 0:
                        pyautogui.keyDown(df_flight["right"][mode_ctrl_state])
                        go_right = 1
                        print('Going Right')

                    # Moving the Aircraft to the Left
                    elif left_x > ctrl_x and go_left == 0:
                        pyautogui.keyDown(df_flight["left"][mode_ctrl_state])
                        go_left = 1
                        print('Going Left')

                    # Keeping Aircraft Straight
                    elif right_x > ctrl_x and left_x < ctrl_x:
                        if go_right == 1:
                            pyautogui.keyUp(df_flight["right"][mode_ctrl_state])
                            go_right = 0
                        if go_left == 1:
                            pyautogui.keyUp(df_flight["left"][mode_ctrl_state])
                            go_left = 0

                    # Missile Launch
                    if abs(r_thmb_y - r_idx_y) < 30 and abs(l_thmb_y - l_idx_y) < 30 and misle_coldwn == 0:
                        misle_coldwn = 1
                        print("Missile Launched")
                        pyautogui.press(df_flight["missiles"][mode_ctrl_state])
                        misle_coldwn_thread.start()

                    # Flares Deploying
                    elif abs(r_thmb_y - r_idx_y) > 30 and abs(l_thmb_y - l_idx_y) < 30 and flare_coldwn == 0:
                        flare_coldwn = 1
                        print("Flares Deployed")
                        pyautogui.press(df_flight["flares"][mode_ctrl_state])
                        flare_coldwn_thread.start()


                    # Aircraft's Gun Fire
                    elif abs(r_thmb_y - r_idx_y) < 30 and abs(l_thmb_y - l_idx_y) > 30 and fire_on == 0:
                        pyautogui.keyDown(df_flight["gun_fire"][mode_ctrl_state])
                        fire_on = 1
                        print("Gun Firing..........")
                    if abs(r_thmb_y - r_idx_y) > 30 and fire_on == 1:
                        pyautogui.keyUp(df_flight["left"][mode_ctrl_state])
                        fire_on = 0
                        print("Firing stopped")

                    # Engaging Speed Boosters
                    if abs(r_idx_x - l_idx_x) < 50 and booster_coldwn == 0:
                        pyautogui.press(df_flight["boosters"][mode_ctrl_state])
                        booster_coldwn = 1
                        print("Booster Engaged")
                        booster_coldwn_thread.start()

                fps, img = fpsReader.update(img, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
                cv2.imshow("Flight", img)

                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0

            #Fruit Ninja
            elif mode == 7:
                hands, img = hand_detect.findHands(img)
                cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
                if hands:
                    lmlist = hands[0]['lmList']
                    ind_x, ind_y = lmlist[8][0], lmlist[8][1]
                    cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)
                    conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, 1920)))
                    conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, 1080)))
                    mouse.move(conv_x, conv_y)
                    fingers = hand_detect.fingersUp(hands[0])
                    if fingers[4] == 1:
                        pyautogui.mouseDown()
                cv2.imshow("Fruit Ninja", img)
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0

            #single hand
            elif mode == 4:
                hand, img = hand_detect.findHands(img)
                if hand and hand[0]["type"] == "Left":
                    fingers = hand_detect.fingersUp(hand[0])
                    totalFingers = fingers.count(1)
                    print(totalFingers)
                    cv2.putText(img, f'Fingers:{totalFingers}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    if totalFingers == 0:
                        pyautogui.keyDown("right")
                        # pyautogui.keyUp('left')
                    if totalFingers == 5:
                        # pyautogui.keyDown("left")
                        pyautogui.keyUp("right")
                cv2.imshow('Single Hand', img)
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0

            #Subway
            elif mode == 2:
                img = cv2.resize(img, (440, 330))
                height, width, channel = img.shape
                width_hf = int(width / 2)
                height_hf = int(height / 2)
                img = pose_detect.findPose(img, draw=False)
                lmList, bboxInfo = pose_detect.findPosition(img, bboxWithHands=False, draw=False)
                # Extracting Shoulder Landmarks
                if lmList:
                    right_x = lmList[11][1] - 7
                    right_y = lmList[11][2]
                    cv2.circle(img, (right_x, right_y), 5, (0, 0, 0), 2)
                    left_x = lmList[12][1] + 7
                    left_y = lmList[12][2]
                    cv2.circle(img, (left_x, left_y), 5, (0, 0, 0), 2)
                    # cv2.line(img, (left_x,left_y), (right_x,right_y), (255, 0, 255), 2)
                    mid_x = left_x + int(abs(right_x - left_x) / 2)
                    mid_y = int(abs(right_y + left_y) / 2)
                    # cv2.circle(img, (mid_x, mid_y), 2, (255, 255, 0), 2)
                    if rec != None:
                        # Sideways movement command
                        if right_x < width_hf and index_pos > 0 and charac_pos[index_pos - 1] == 0:
                            charac_pos[index_pos] = 0
                            charac_pos[index_pos - 1] = 1
                            pyautogui.press(df_sub["left"][mode_ctrl_state])
                            index_pos -= 1
                            print("Left key")
                            print(charac_pos)
                        if left_x > width_hf and index_pos < 2 and charac_pos[index_pos + 1] == 0:
                            print("Right key")
                            charac_pos[index_pos] = 0
                            charac_pos[index_pos + 1] = 1
                            pyautogui.press(df_sub["right"][mode_ctrl_state])
                            index_pos += 1
                            print(charac_pos)
                        if right_x > width_hf and left_x < width_hf and index_pos == 0:
                            charac_pos[index_pos] = 0
                            charac_pos[index_pos + 1] = 1
                            index_pos += 1
                            pyautogui.press(df_sub["right"][mode_ctrl_state])
                            print(charac_pos)
                            print('left to center')
                        if right_x > width_hf and left_x < width_hf and index_pos == 2:
                            charac_pos[index_pos] = 0
                            charac_pos[index_pos - 1] = 1
                            index_pos -= 1
                            pyautogui.press(df_sub["left"][mode_ctrl_state])
                            print('right to center')
                            print(charac_pos)
                hands, img = hand_detect.findHands(img, draw=True, flipType=False)
                if len(hands) == 2:
                    if hands[0]['type'] == "Left":
                        hands_l, hands_r = hands[0], hands[1]
                    else:
                        hands_l, hands_r = hands[1], hands[0]

                    lmlist_l, lmlist_r = hands_l['lmList'], hands_r['lmList']

                    fingers_left = hand_detect.fingersUp(hands_l)
                    fingers_right = hand_detect.fingersUp(hands_r)  # Command to Start the game

                    # print(fingers_left,fingers_right)
                    if fingers_right.count(1) == 3 and fingers_left.count(1) == 3 and fingers_right[1] == 1 and fingers_right[
                        2] == 1 and fingers_left[1] == 1 and fingers_left[1] == 1:
                        fixedx = left_x + int(abs(right_x - left_x) / 2)
                        fixedy = int(abs(right_y + left_y) / 2)
                        rec = 35
                        pyautogui.press('space')

                    # Up and Down command
                if fixedy is not None:
                    if (mid_y - fixedy) <= -24:
                        pyautogui.press(df_sub["up"][mode_ctrl_state])
                        print('jump')
                    elif (mid_y - fixedy) >= 40:
                        pyautogui.press(df_sub["down"][mode_ctrl_state])
                        print('down')
                center_arrow = 10
                cv2.circle(img, (width_hf, height_hf), 2, (0, 255, 255), 2)
                cv2.line(img, (width_hf, height_hf - center_arrow), (width_hf, height_hf + center_arrow), (0, 255, 0), 2)
                cv2.line(img, (width_hf - center_arrow, height_hf), (width_hf + center_arrow, height_hf), (0, 255, 0), 2)
                # Lines to be crossed to detect up and down movement
                # if rec is not None:
                #     cv2.line(img, (0, fixedy), (width, fixedy), (0, 0, 0), 2)
                #     cv2.line(img, (0, fixedy - 24), (width, fixedy - 24), (0, 0, 0), 2)
                #     cv2.line(img, (0, fixedy + rec), (width, fixedy + rec), (0, 0, 0), 2)

                cv2.imshow('Subway Surfers', img)
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0
            # jump
            elif mode == 6:
                img = cv2.resize(img, (440, 330))
                height, width, channel = img.shape
                width_hf = int(width / 2)
                height_hf = int(height / 2)
                img = pose_detect.findPose(img, draw=False)
                lmList, bboxInfo = pose_detect.findPosition(img, bboxWithHands=False, draw=False)
                # Extracting Shoulder Landmarks
                if lmList:
                    right_x = lmList[11][1] - 7
                    right_y = lmList[11][2]
                    cv2.circle(img, (right_x, right_y), 5, (0, 0, 0), 2)
                    left_x = lmList[12][1] + 7
                    left_y = lmList[12][2]
                    cv2.circle(img, (left_x, left_y), 5, (0, 0, 0), 2)
                    mid_x = left_x + int(abs(right_x - left_x) / 2)
                    mid_y = int(abs(right_y + left_y) / 2)

                hands, img = hand_detect.findHands(img, draw=True, flipType=False)
                if len(hands) == 2:
                    if hands[0]['type'] == "Left":
                        hands_l, hands_r = hands[0], hands[1]
                    else:
                        hands_l, hands_r = hands[1], hands[0]

                    lmlist_l, lmlist_r = hands_l['lmList'], hands_r['lmList']

                    fingers_left = hand_detect.fingersUp(hands_l)
                    fingers_right = hand_detect.fingersUp(hands_r)  # Command to Start the game

                    # print(fingers_left,fingers_right)
                    if fingers_right.count(1) == 3 and fingers_left.count(1) == 3 and fingers_right[1] == 1 and \
                            fingers_right[
                                2] == 1 and fingers_left[1] == 1 and fingers_left[1] == 1:
                        fixedx = left_x + int(abs(right_x - left_x) / 2)
                        fixedy = int(abs(right_y + left_y) / 2)
                        rec = 35
                        pyautogui.press('space')

                    # Up and Down command
                if fixedy is not None:
                    if (mid_y - fixedy) <= -24:
                        pyautogui.press(df_sub["up"][mode_ctrl_state])
                        print('jump')
                    elif (mid_y - fixedy) >= 40:
                        pyautogui.press(df_sub["down"][mode_ctrl_state])
                        print('down')

                center_arrow = 10
                cv2.circle(img, (width_hf, height_hf), 2, (0, 255, 255), 2)
                cv2.line(img, (width_hf, height_hf - center_arrow), (width_hf, height_hf + center_arrow), (0, 255, 0), 2)
                cv2.line(img, (width_hf - center_arrow, height_hf), (width_hf + center_arrow, height_hf), (0, 255, 0), 2)
                # Lines to be crossed to detect up and down movement
                # if rec is not None:
                #     cv2.line(img, (0, fixedy), (width, fixedy), (0, 0, 0), 2)
                #     cv2.line(img, (0, fixedy - 24), (width, fixedy - 24), (0, 0, 0), 2)
                #     cv2.line(img, (0, fixedy + rec), (width, fixedy + rec), (0, 0, 0), 2)

                cv2.imshow('Jump', img)
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0
            # Dual hand
            elif mode == 1:
                hands, img = hand_detect.findHands(img, draw=True, flipType=False)
                if len(hands) == 2:
                    if hands[0]['type'] == "Left":
                        hands_l, hands_r = hands[0], hands[1]
                    else:
                        hands_l, hands_r = hands[1], hands[0]
                    l_fing_upno = hand_detect.fingersUp(hands_l)
                    r_fing_upno = hand_detect.fingersUp(hands_r)
                    if (l_fing_upno.count(1) == 4) and left_hand_op == 0:
                        left_hand_op = 1
                        left_dual_thread.start()
                        print("left open")
                    if (l_fing_upno.count(0) == 4) and left_hand_op == 1:
                        pyautogui.keyUp("left")
                        left_hand_op = 0
                    if (r_fing_upno.count(1) == 4) and right_hand_op == 0:
                        right_hand_op = 1
                        right_dual_thread.start()
                        print("right open")
                    if (r_fing_upno.count(0) == 4) and right_hand_op == 1:
                        pyautogui.keyUp("right")
                        right_hand_op = 0

                fps, img = fpsReader.update(img, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
                cv2.imshow("Dual Hands", img)

                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status =0

            # Racing
            elif mode == 5:
                hands, img = hand_detect.findHands(img, draw=True)
                if len(hands) == 2:
                    if hands[0]['type'] == "Left":
                        hands_l, hands_r = hands[0], hands[1]
                    else:
                        hands_l, hands_r = hands[1], hands[0]

                    lmlist_l, lmlist_r = hands_l['lmList'], hands_r['lmList']

                    r_sp_x, r_sp_y = lmlist_r[9][0], lmlist_r[9][1]
                    l_sp_x, l_sp_y = lmlist_l[9][0], lmlist_l[9][1]
                    r_idx_x, r_idx_y = lmlist_r[6][0], lmlist_r[6][1]
                    l_idx_x, l_idx_y = lmlist_l[6][0], lmlist_l[6][1]
                    r_thmb_x, r_thmb_y = lmlist_r[4][0], lmlist_r[4][1]
                    l_thmb_x, l_thmb_y = lmlist_l[4][0], lmlist_l[4][1]
                    cen_x = l_sp_x + (r_sp_x - l_sp_x) // 2
                    cen_y = (r_sp_y + l_sp_y) // 2

                    cv2.circle(img, (r_sp_x, r_sp_y), 5, (0, 255, 255), 2)
                    cv2.circle(img, (l_sp_x, l_sp_y), 5, (0, 255, 255), 2)
                    cv2.circle(img, (cen_x, cen_y), 5, (0, 255, 255), 5)
                    # cv2.line(img,(cen_x, 0), (cen_x,1000), (0,0,0),5)
                    # cv2.line(img,(0, cen_y), (1000, cen_y), (0,0,0),5)
                    cv2.line(img, (r_sp_x, r_sp_y), (l_sp_x, l_sp_y), (0, 0, 0), 5)
                    # print(abs(r_sp_y-cen_y),abs(l_sp_y-cen_y))
                    distance = math.sqrt((r_sp_x - l_sp_x) ** 2 + (r_sp_y - l_sp_y) ** 2)
                    # print(distance)

                    if distance <= 260:
                        cv2.circle(img, (cen_x, cen_y), 80, (0, 255, 255), 5)
                        # Left Steer
                        if abs(r_sp_y - cen_y) > 40 and r_sp_y < cen_y and left_steer_on == 0:
                            left_steer_on = 1
                            strght_steer_on = 0
                            left_steer_thread.start()
                        # Right Steer
                        if abs(r_sp_y - cen_y) > 40 and r_sp_y > cen_y and right_steer_on == 0:
                            print("Right steer")
                            right_steer_on = 1
                            strght_steer_on = 0
                            right_steer_thread.start()
                        # Going Straight
                        if abs(r_sp_y - cen_y) < 40 and strght_steer_on == 0:
                            right_steer_on = 0
                            left_steer_on = 0
                            strght_steer_on = 1
                            strght_steer_thread.start()


                        # nitros Deploying
                        elif abs(r_thmb_y - r_idx_y) > 30 and abs(l_thmb_y - l_idx_y) < 30 and nitro_coldwn == 0:
                            nitro_coldwn = 1
                            print("nitros Deployed")
                            pyautogui.press("Space")
                            nitro_coldwn_thread.start()

                        elif abs(r_thmb_y - r_idx_y) < 30 and abs(l_thmb_y - l_idx_y) > 30 and brake_on == 0:
                            pyautogui.keyDown("down")
                            brake_on = 1
                            print("Braking")
                        if abs(r_thmb_y - r_idx_y) > 30 and brake_on == 1:
                            pyautogui.keyUp("down")
                            brake_on = 0
                            print("Going")

               # fps, img = fpsReader.update(img, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
                cv2.imshow("Racing", img)

                if cv2.waitKey(1) & 0xFF == ord('o'):
                    exit_status = 0
        if exit_status ==0:
            cap.release()
            cv2.destroyAllWindows()
            break
