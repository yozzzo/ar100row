import cv2
import numpy as np

# トラッキング対象の画像を読み込み
target_image = cv2.imread('marker.jpg')
target_image = cv2.resize(target_image, (target_image.shape[1] // 2, target_image.shape[0] // 2))

# SIFTアルゴリズムを使用してキーポイントとディスクリプタを抽出
sift = cv2.SIFT_create(nfeatures=500)
kp1, des1 = sift.detectAndCompute(target_image, None)
des1 = np.float32(des1)  # ディスクリプタの型をfloat32に変換

# FLANNベースのマッチャーを作成
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# カメラキャプチャ
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 動画フレームのキーポイントとディスクリプタを抽出
    kp2, des2 = sift.detectAndCompute(frame, None)
    des2 = np.float32(des2)  # ディスクリプタの型をfloat32に変換

    # キーポイント間のマッチングを行い、ホモグラフィを計算
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # トラッキング対象の画像の四隅を計算
        h, w = target_image.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
