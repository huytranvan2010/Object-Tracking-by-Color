# USAGE
# python tracking_object_by_color.py --video BallTracking.mp4

import imutils
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

# Xác định khoảng màu (color ranges)
# Để xác định được khoảng màu này cần xem trước màu vật thể và xem bảng màu
# Mỗi khoảng màu tương ứng với một object
colorRanges = [
    ((29, 86, 6), (64, 255, 255), "green"),
    ((57, 68, 0), (151, 255, 255), "blue")
]

# Nếu a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):    # args ở dạng dict
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

# Duyệt qua các frame
while True:     # cứ hết câu lệnh ở cuối mới chuyển lên frame sau, ở trong while cho 1 frame duy nhất
    # grab the current frame
    ret, frame = camera.read()

    # If we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get("video") and not ret:   # ko trả về frame thì ret = False, cái này ko cần cho webcam
        break

    # resize, blur and convert frame to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)      # kernel size = 11x11, 3rd argument = 0 (sifmaX, sigmaY) tự tính theo kernel
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) 

    # Duyệt qua các khoảng màu
    for (lower, upper, colorName) in colorRanges:
        # xây dựng a binary mask dựa trên khoảng màu HSV, sau đó thực hiện erosion (xói mòn) và dilation(giãn nở) để loại
        # bỏ các small blobs
        mask = cv2.inRange(hsv, lower, upper)   # trả về binary mask (pixel nào trong khoảng màu -> 255 - white, nằm ngoài -> 0 - black)
        mask = cv2.erode(mask, None, iterations=2)  # iterations là số lần áp dụng erosion
        mask = cv2.dilate(mask, None, iterations=2)

        # tìm contours trong mask 
        # do duyệt qua từng color range (tương ứng với từng vật thể) nên chính là tìm contours cho từng object
        ctns = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctns = imutils.grab_contours(ctns)

        # chỉ thực hiện nếu có ít nhất 1 contours được tìm thất cho 1 mask
        if len(ctns) > 0:
            # tìm contour lớn nhất trong mask, sau đó sử dụng nó để xác định enclosing circle - vòng tròn bao quanh
            # và centroid
            c = max(ctns, key=cv2.contourArea)      # key chỉ nhập hàm ko cần truyền đối số

            # trả về coordinates of the center and radius
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # tính moment của contour và tính coordinate of centroid của contour
            M = cv2.moments(c)
            (cX, cY) = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            # Chỉ vẽ the enclosing circle và text nếu radius thỏa mãn điều kiện
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

# clearnup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
