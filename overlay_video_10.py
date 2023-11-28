import cv2
import cv2.aruco as aruco
import numpy as np
import time

def click_event(event, x, y, flags, param):
    global play_video
    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 <= x <= 150 and 50 <= y <= 150:  # スタートボタンの位置
            play_video = True

# カメラデバイスと動画ファイルを開く
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')  # 動画ファイルのパスを設定

# スタートボタンと背景画像を読み込む
# start_button = cv2.imread('start_button.jpg')
background = cv2.imread('back.jpg')
logo = cv2.imread('start_button.png')  # ロゴの画像を読み込む

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

play_video = False
last_tracking_time = None
tracking_lost = False

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        corner = corners[0][0]
        center = np.mean(corner, axis=0)

        # 半透明の背景画像をトラッキング
        overlay = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # ロゴの配置
        logo_height, logo_width = logo.shape[:2]
        x_center = frame.shape[1] // 2 - logo_width // 2
        y_center = frame.shape[0] // 2 - logo_height // 2
        frame[y_center:y_center+logo_height, x_center:x_center+logo_width] = logo

        if play_video:
            # 動画の次のフレームを取得
            ret_video, overlay_frame = video.read()
            if not ret_video:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, status = cv2.findHomography(pts_src, corner)
            temp = cv2.warpPerspective(overlay_frame, h, (frame.shape[1], frame.shape[0]))
            np.copyto(frame, temp, where=temp>0)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
