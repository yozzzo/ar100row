import cv2
import numpy as np

# トラッキング対象の画像と動画の読み込み
marker_image = cv2.imread('marker_raw.jpg')
# marker_image = cv2.resize(marker_image, (marker_image.shape[1] // 3, marker_image.shape[0] // 3))
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')

# SIFTディテクタを作成
sift = cv2.SIFT_create(nfeatures=100)

# トラッキング対象画像のキーポイントとディスクリプタを計算
kp1, des1 = sift.detectAndCompute(marker_image, None)

# マッチャーを作成（FLANNマッチャーを使用）
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームのキーポイントとディスクリプタを計算
    kp2, des2 = sift.detectAndCompute(frame, None)

    # マッチング
    matches = flann.knnMatch(des1, des2, k=2)

    # デービス・ロウの比率テストを使用して良いマッチを保持
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # トラッキング対象の位置を特定
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ホモグラフィを計算
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 動画のフレームを取得し、ホモグラフィを適用してオーバーレイ
        ret_video, overlay_frame = video.read()
        if ret_video:
            h, w, _ = marker_image.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            frame = cv2.warpPerspective(overlay_frame, M, (frame.shape[1], frame.shape[0]), frame, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
