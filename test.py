# import imutils
from cv2 import cv2
import numpy as np
warped_img = None
input_img2 = cv2.imread("topview.jpg")
input_img1 = cv2.resize(input_img2, (900, 700))
# cv2.imshow("1", input_img1)
# cv2.waitKey(0)

input_img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)

# input_img = cv2.GaussianBlur(cv2.cvtColor(
#     input_img1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
# cv2.imwrite("D:\CiPHER\working\grayBlur.jpg", input_img)
# input_img = input_img1

cv2.imshow("11", input_img)
cv2.waitKey(0)


(thresh, blackAndWhiteImage1) = cv2.threshold(
    input_img, 194, 202, cv2.THRESH_BINARY)

cv2.imshow("blackAndWhiteImage1", blackAndWhiteImage1)
cv2.waitKey(0)

# (thresh, blackAndWhiteImage2) = cv2.threshold(
#     input_img, 153, 156, cv2.THRESH_BINARY)

# cv2.imshow("blackAndWhiteImage2", blackAndWhiteImage2)
# cv2.waitKey(0)

# blackAndWhiteImage = blackAndWhiteImage1+blackAndWhiteImage2

# cv2.imshow("blackAndWhiteImage", blackAndWhiteImage)
# cv2.waitKey(0)

contours, _ = cv2.findContours(
    blackAndWhiteImage1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h, w = blackAndWhiteImage1.shape
# cv2.drawContours(input_img1, contours, -1, (0, 255, 0), 3)
# cv2.imshow("Mask", input_img1)
# cv2.waitKey(0)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if(area < (h*w) and area > ((h/3)*(w/3))):
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        x2 = approx.ravel()[2]
        y2 = approx.ravel()[3]
        x3 = approx.ravel()[4]
        y3 = approx.ravel()[5]
        x4 = approx.ravel()[6]
        y4 = approx.ravel()[7]
        break
# Detecting which point is at which corner of image frame
if(x1 < w/2 and y1 < h/2):
    x11, y11 = 0, 0
elif(x1 > w/2 and y1 < h/2):
    x11, y11 = w, 0
elif(x1 > w/2 and y1 > h/2):
    x11, y11 = w, h
elif(x1 < w/2 and y1 > h/2):
    x11, y11 = 0, h
if(x2 < w/2 and y2 < h/2):
    x22, y22 = 0, 0
elif(x2 > w/2 and y2 < h/2):
    x22, y22 = w, 0
elif(x2 > w/2 and y2 > h/2):
    x22, y22 = w, h
elif(x2 < w/2 and y2 > h/2):
    x22, y22 = 0, h
if(x3 < w/2 and y3 < h/2):
    x33, y33 = 0, 0
elif(x3 > w/2 and y3 < h/2):
    x33, y33 = w, 0
elif(x3 > w/2 and y3 > h/2):
    x33, y33 = w, h
elif(x3 < w/2 and y3 > h/2):
    x33, y33 = 0, h
if(x4 < w/2 and y4 < h/2):
    x44, y44 = 0, 0
elif(x4 > w/2 and y4 < h/2):
    x44, y44 = w, 0
elif(x4 > w/2 and y4 > h/2):
    x44, y44 = w, h
elif(x4 < w/2 and y4 > h/2):
    x44, y44 = 0, h
# Applying Warp Perspective
pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
pts2 = np.float32([[x11, y11], [x22, y22], [x33, y33], [x44, y44]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped_img = cv2.warpPerspective(input_img, matrix, (w, h))
warped_img1 = cv2.warpPerspective(input_img1, matrix, (w, h))
cv2.imshow("warped_img", warped_img)
# cv2.imwrite("D:\CiPHER\originalImg.jpg", warped_img)
cv2.waitKey(0)

# (thresh, blackAndWhiteImage4) = cv2.threshold(
#     warped_img, 151, 160, cv2.THRESH_BINARY_INV)

# cv2.imshow("blackAndWhiteImage4", blackAndWhiteImage4)
# cv2.waitKey(0)

(thresh, blackAndWhiteImage3) = cv2.threshold(
    warped_img, 151, 202, cv2.THRESH_BINARY_INV)

# blackAndWhiteImage3 = blackAndWhiteImage3+blackAndWhiteImage4

cv2.imshow("blackAndWhiteImage3", blackAndWhiteImage3)
cv2.waitKey(0)

contours, _ = cv2.findContours(
    blackAndWhiteImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h, w = blackAndWhiteImage3.shape

for cnt in contours:
    area = cv2.contourArea(cnt)
    if(area < (h*w) and area > 500):

        # cnts = imutils.grab_contours(cnt)
        # c = max(cnts, key=cv2.contourArea)
        # print(cnt)
        # left = tuple(cnt[cnt[:, :, 0].argmin()][0])
        # right = tuple(cnt[cnt[:, :, 0].argmax()][0])
        # top = tuple(cnt[cnt[:, :, 1].argmin()][0])
        # bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # print("Max: ", right)
        # cv2.circle(warped_img1, left, 2, (0, 50, 255), -1)
        # cv2.circle(warped_img1, right, 2, (0, 255, 255), -1)
        # cv2.circle(warped_img1, top, 2, (255, 50, 0), -1)
        # cv2.circle(warped_img1, bottom, 2, (255, 255, 0), -1)

        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # if len(approx) != 4:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # cv2.circle(warped_img1, (cx, cy), 8, (0, 255, 255), -1)
        print("Area: ", area)
        print("(X, Y): ", (cx, cy))
        print(len(approx))
        print("")

cv2.drawContours(warped_img1, contours, -1, (0, 255, 0), 3)
cv2.imshow("Mask", warped_img1)
cv2.waitKey(0)
