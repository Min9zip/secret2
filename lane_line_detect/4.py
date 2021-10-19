import cv2
import numpy as np


capture = cv2.VideoCapture('videos/challenge.mp4')


if capture.isOpened() == False:
   print("카메라를 열 수 없습니다.")
   exit(1)

while True:
   ret, img_src = capture.read()
   img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

   lower_yellow = (15, 100, 100)
   upper_yellow = (40, 255, 255)

   yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
   img_yellow = cv2.bitwise_and(img_src, img_src, mask=yellow_mask)

   lower_white = (150, 150, 150)
   upper_white = (255, 255, 255)
   white_mask = cv2.inRange(img_src, lower_white, upper_white)
   img_white = cv2.bitwise_and(img_src, img_src, mask=white_mask)




   img_lane = cv2.addWeighted(img_white, 1., img_yellow, 1., 0)

   img_gray = cv2.cvtColor(img_lane,cv2.COLOR_BGR2GRAY)
   img_gray = cv2.GaussianBlur(img_gray,(5,5),0)
   _,img_binary = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

   img_canny= cv2.Canny(img_binary,50,150)
   img_mask = np.zeros_like(img_canny)
   height, width = img_mask.shape[:2]

   trap_bottom_width = 0.8
   trap_top_width = 0.1
   trap_height = 0.4

   mask_color = (255, 255, 255)

   pts = np.array([[
      ((width * (1 - trap_bottom_width)) // 2, height),
      ((width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
      (width - (width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
      (width - (width * (1 - trap_bottom_width)) // 2, height)]],
      dtype=np.int32)

   cv2.fillPoly(img_mask, pts, mask_color)
   img_canny = cv2.bitwise_and(img_canny, img_mask)

   img_roi =img_canny

   rho= 2
   theta = 1 * np.pi /180
   threshold = 15
   min_line_length = 10
   max_line_gap = 20
   lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,minLineLength=min_line_length,
                           maxLineGap=max_line_gap)

   for i, line in enumerate(lines):
      cv2.line(img_src, (line[0][0], line[0][1]),
               (line[0][2], line[0][3]), (0, 255, 0), 2)






   if ret == False: # 동영상 끝까지 재생
       print("동영상 읽기 완료")
       break
   # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
   if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
       capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
   cv2.imshow('Video-dst', img_src)
   key = cv2.waitKey(25) # 33ms마다
   if key == 27:         # Esc 키
       break

capture.release()
cv2.destroyAllWindows()