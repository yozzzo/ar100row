import cv2
import cv2.aruco as aruco
import numpy as np

targetVideo = 0 # カメラデバイス
cap = cv2.VideoCapture(targetVideo)

# 表示する画像を読み込む
overlay_image = cv2.imread('sample.jpg')  # 画像ファイルのパスを設定
overlay_height, overlay_width = overlay_image.shape[:2]

while cap.isOpened():
    ret, img = cap.read()
    if img is None:
        break

    # ARマーカーを検出
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # ARマーカーが見つかった場合の処理
    if len(corners) > 0:
        # マーカーの中心座標を計算
        for corner in corners:
            center_x = int(corner[0][0][0] + corner[0][2][0]) // 2
            center_y = int(corner[0][0][1] + corner[0][2][1]) // 2

            # 画像をマーカーの位置にオーバーレイ
            x_start = center_x - overlay_width // 2
            x_end = center_x + overlay_width // 2
            y_start = center_y - overlay_height // 2
            y_end = center_y + overlay_height // 2

            # 画像をフレームに合成
            if x_start >= 0 and y_start >= 0 and x_end < img.shape[1] and y_end < img.shape[0]:
                img[y_start:y_end, x_start:x_end] = overlay_image

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
