import cv2
import cv2.aruco as aruco
import numpy as np

targetVideo = 0 # カメラデバイス
cap = cv2.VideoCapture( targetVideo )

overlay_image = cv2.imread('sample.jpg')  # 画像ファイルのパスを設定
overlay_height, overlay_width = overlay_image.shape[:2]

# 立方体の座標
cube = np.float32([[0.025,-0.025,0], [-0.025,0.025,0], [0.025,0.025,0], [-0.025,-0.025,0],
                    [0.025,-0.025,0.05], [-0.025,0.025,0.05], [0.025,0.025,0.05], [-0.025,-0.025,0.05]
                    ])

while cap.isOpened():
    ret, img = cap.read()
    if img is None :
        break
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    overlay_image = cv2.resize(overlay_image, dim, interpolation = cv2.INTER_AREA)
    # Check if frame is not empty
    if not ret:
        continue
    # Set AR Marker
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    center = (width, height)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
            )
    dist_coeffs = np.zeros((4,1)) # レンズ歪みなしの設定
    if len(corners) > 0:
        for i, corner in enumerate(corners):
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
            
            tvec = np.squeeze(tvecs)
            rvec = np.squeeze(rvecs)
            rvec_matrix = cv2.Rodrigues(rvec) # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = rvec_matrix[0]      # rodoriguesから抜き出し

            # cubeの座標をARマーカに合わせる
            imgpts, jac = cv2.projectPoints(cube, rvecs, tvecs, camera_matrix, dist_coeffs)
            overlay_imgpts, jac = cv2.projectPoints(overlay_image, rvecs, tvecs, camera_matrix, dist_coeffs)
            outpts = []
            for lp in imgpts:
                lp_int = lp.astype(np.int64)
                outpts.append( tuple(lp_int.ravel()) )

            # ARマーカに合わせたcube座標を描画:底面
            cv2.line(img,outpts[0],outpts[2],(255,0,0),10)
            cv2.line(img,outpts[1],outpts[2],(255,0,0),10)
            cv2.line(img,outpts[1],outpts[3],(255,0,0),10)
            cv2.line(img,outpts[3],outpts[0],(255,0,0),10)
            img[outpts[0][1]:outpts[2][1],outpts[0][0]:outpts[1][0]] = overlay_image

            # ARマーカに合わせたcube座標を描画:上面
            cv2.line(img,outpts[4],outpts[6],(0,255,0),10)
            cv2.line(img,outpts[5],outpts[6],(0,255,0),10)
            cv2.line(img,outpts[5],outpts[7],(0,255,0),10)
            cv2.line(img,outpts[7],outpts[4],(0,255,0),10)

            # ARマーカに合わせたcube座標を描画:支柱
            cv2.line(img,outpts[0],outpts[4],(0,0,250),10)
            cv2.line(img,outpts[1],outpts[5],(0,0,250),10)
            cv2.line(img,outpts[2],outpts[6],(0,0,250),10)
            cv2.line(img,outpts[3],outpts[7],(0,0,250),10)

            # ARマーカに合わせたcube座標を描画
            cv2.circle(img, outpts[0], 10, (0, 240, 160), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '0', outpts[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[1], 10, (40, 200, 200), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '1', outpts[1], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[2], 10, (80, 160, 240), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '2', outpts[2], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[3], 10, (120, 120, 40), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '3', outpts[3], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[4], 10, (160, 80, 80), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '4', outpts[4], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[5], 10, (200, 40, 120), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '5', outpts[5], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[6], 10, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '6', outpts[6], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

            cv2.circle(img, outpts[7], 10, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img, '7', outpts[7], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

