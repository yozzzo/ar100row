import cv2
import cv2.aruco as aruco
import numpy as np
import time

# カメラデバイスと動画ファイルを開く
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')  # 動画ファイルのパスを設定

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

last_tracking_time = None
tracking_lost = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 動画の次のフレームを取得
    ret_video, overlay_frame = video.read()
    if not ret_video:
        # 動画の最初に戻る
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        corner = corners[0][0]
        pts_dst = np.array([corner[0], corner[1], corner[2], corner[3]])
        pts_src = np.array([[0, 0], [overlay_frame.shape[1], 0], [overlay_frame.shape[1], overlay_frame.shape[0]], [0, overlay_frame.shape[0]]], dtype=float)

        # 動画のサイズをマーカーサイズの110%に設定
        scale = 1.1
        pts_src *= scale

        h, status = cv2.findHomography(pts_src, pts_dst)
        temp = cv2.warpPerspective(overlay_frame, h, (frame.shape[1], frame.shape[0]))
        np.copyto(frame, temp, where=temp>0)

        last_tracking_time = time.time()
        tracking_lost = False
    elif tracking_lost or (last_tracking_time is not None and time.time() - last_tracking_time > 0.8):
        # トラッキングが外れた場合、または0.8秒以上再トラッキングできない場合
        frame[0:overlay_frame.shape[0], 0:overlay_frame.shape[1]] = overlay_frame
        tracking_lost = True

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
