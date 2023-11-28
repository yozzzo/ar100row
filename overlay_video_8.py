import cv2
import cv2.aruco as aruco
import numpy as np
import time

# カメラデバイスと動画ファイルを開く
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')  # 動画ファイルのパスを設定

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

play_video = False
last_tracking_time = None
tracking_lost = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if not play_video:
        # 半透明のグレー背景と再生ボタンのテキストを表示
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (100, 100, 100), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "Press Space to Play", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        # 動画の次のフレームを取得
        ret_video, overlay_frame = video.read()
        if not ret_video:
            # 動画の最初に戻る
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            corner = corners[0][0]
            center = np.mean(corner, axis=0)

            # 動画のサイズをマーカーサイズの110%に設定
            scale = 1.1
            overlay_center = np.array([overlay_frame.shape[1] / 2, overlay_frame.shape[0] / 2])
            pts_src = np.array([[0, 0], [overlay_frame.shape[1], 0], [overlay_frame.shape[1], overlay_frame.shape[0]], [0, overlay_frame.shape[0]]], dtype=float) - overlay_center
            pts_src = pts_src * scale + center

            h, status = cv2.findHomography(pts_src, corner)
            temp = cv2.warpPerspective(overlay_frame, h, (frame.shape[1], frame.shape[0]))
            np.copyto(frame, temp, where=temp>0)

            last_tracking_time = time.time()
            tracking_lost = False

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        play_video = True

cap.release()
video.release()
cv2.destroyAllWindows()
