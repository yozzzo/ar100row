import cv2
import numpy as np

# トラッキング対象の画像と動画の読み込み
marker_image = cv2.imread('marker.jpg')
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')

# ORBディテクタを作成
orb = cv2.ORB_create()

# トラッキング対象画像のキーポイントとディスクリプタを計算
kp1, des1 = orb.detectAndCompute(marker_image, None)

# マッチャーを作成（FLANNマッチャーも使用可能）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームのキーポイントとディスクリプタを計算
    kp2, des2 = orb.detectAndCompute(frame, None)

    # マッチング
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # トラッキング対象の位置を特定
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

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
