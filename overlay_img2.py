import cv2
import cv2.aruco as aruco
import numpy as np

targetVideo = 0 # カメラデバイス
cap = cv2.VideoCapture(targetVideo)

overlay_image = cv2.imread('sample.jpg')  # 画像ファイルのパスを設定

# ARマーカーの設定
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ARマーカーの検出
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        # ARマーカーの領域を取得
        corner = corners[0][0]
        pts_dst = np.array([corner[0], corner[1], corner[2], corner[3]])
        pts_src = np.array([[0, 0], [overlay_image.shape[1], 0], [overlay_image.shape[1], overlay_image.shape[0]], [0, overlay_image.shape[0]]], dtype=float)

        # 画像の変換行列を計算
        h, status = cv2.findHomography(pts_src, pts_dst)

        # 画像をワーピングしてフレームに貼り付け
        temp = cv2.warpPerspective(overlay_image, h, (frame.shape[1], frame.shape[0]))
        np.copyto(frame, temp, where=temp>0)

    # 結果を表示
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
