from os.path import sep
import cv2 as cv2

# load the overlay image. size should be smaller than video frame size
# img = cv2.VideoCapture('photos' + sep + 'Baslksz-3.mp4')
video_name = 'video.mp4'
img = cv2.VideoCapture(video_name)

# Get Image dimensions
img.set(cv2.CAP_PROP_FRAME_WIDTH, 150)  # float `width`
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)
width = 150
height = 150

# Start Capture
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# frame_vid = img.read()

# Decide X,Y location of overlay image inside video frame.
# following should be valid:
#   * image dimensions must be smaller than frame dimensions
#   * x+img_width <= frame_width
#   * y+img_height <= frame_height
# otherwise you can resize image as part of your code if required

x = 50
y = 50

video_frame_counter = 0

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        ret_video, frame_video = img.read()
        video_frame_counter += 1

        if video_frame_counter == img.get(cv2.CAP_PROP_FRAME_COUNT):
            video_frame_counter = 0
            img.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if ret_video:
            # add image to frame
            frame_video = cv2.resize(frame_video, (width, height))
            frame[y:y + width, x:x + height] = frame_video

            '''
            tr = 0.3 # transparency between 0-1, show camera if 0
            frame = ((1-tr) * frame.astype(np.float) + tr * frame_vid.astype(np.float)).astype(np.uint8)
            '''
            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Exit if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

img.release()
cap.release()
cv2.destroyAllWindows()