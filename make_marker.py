import cv2
import cv2.aruco as aruco
import numpy as np

# 写真の読み込み
photo = cv2.imread('sample.jpg')  # 写真のパスを指定

# ArUcoマーカーの生成
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker = aruco.drawMarker(aruco_dict, 2, 700)  # マーカーIDとサイズを指定

# マーカーを3チャンネル（カラー）形式に変換
marker_colored = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

# 写真にマーカーを重ねる（位置やサイズは調整が必要）
photo[100:800, 100:800] = marker_colored

# 結果を表示
cv2.imshow('AR Marker on Photo', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
