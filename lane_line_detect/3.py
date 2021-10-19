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




   img_mask = np.zeros_like(img_lane)
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
   img_dst = cv2.bitwise_and(img_lane, img_mask)










   if ret == False: # 동영상 끝까지 재생
       print("동영상 읽기 완료")
       break
   # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
   if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
       capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
   cv2.imshow('Video-dst', img_dst)
   key = cv2.waitKey(25) # 33ms마다
   if key == 27:         # Esc 키
       break

capture.release()
cv2.destroyAllWindows()