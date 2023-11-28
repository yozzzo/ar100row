import cv2
import cv2.aruco as aruco
import numpy as np

# カメラデバイスと動画ファイルを開く
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')  # 動画ファイルのパスを設定

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

prev_center = None
shake_offset = 10  # 手ブレのオフセット

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
        center = np.mean(corner, axis=0)

        if prev_center is not None:
            movement = center - prev_center
            movement = np.clip(movement, -shake_offset, shake_offset)
            corner += movement

        prev_center = center
        pts_dst = np.array([corner[0], corner[1], corner[2], corner[3]])
        pts_src = np.array([[0, 0], [overlay_frame.shape[1], 0], [overlay_frame.shape[1], overlay_frame.shape[0]], [0, overlay_frame.shape[0]]], dtype=float)

        # 動画のサイズを若干小さくする
        scale = 1.01
        pts_src *= scale

        h, status = cv2.findHomography(pts_src, pts_dst)
        temp = cv2.warpPerspective(overlay_frame, h, (frame.shape[1], frame.shape[0]))
        np.copyto(frame, temp, where=temp>0)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
