import cv2
import numpy as np

def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
   # 예외처리
   if lines is None:
       return
   if len(lines) == 0:
       return
   draw_right = True
   draw_left = True
   # 모든 선의 기울기 찾기
   # 기울기의 절대값이 임계값 보다 커야 추출됨
   slope_threshold = 0.5
   slopes = []
   new_lines = []
   for line in lines:
       x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
       # 기울기 계산
       if x2 - x1 == 0.:  # 기울기의 분모가 0일때 예외처리
           slope = 999.  # practically infinite slope
       else:
           slope = (y2 - y1) / (x2 - x1)
       # 조건을 만족하는 line을 new_lines에 추가
       if abs(slope) > slope_threshold:
           slopes.append(slope)
           new_lines.append(line)
   lines = new_lines
   # 라인을 오른쪽과 왼쪽으로 구분하기
   # 기울기 및 점의 위치가 영상의 가운데를 기준으로 왼쪽 또는 오른쪽에 위치
   right_lines = []
   left_lines = []
   for i, line in enumerate(lines):
       x1, y1, x2, y2 = line[0]
       img_x_center = img.shape[1] / 2  # 영상의 가운데 x 좌표
       #기울기 방향이 바뀐이유는 y축의 방향이 아래로 내려오기 때문
       if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
           right_lines.append(line)
       elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
           left_lines.append(line)

   # np.polyfit()함수를 사용하기 위해 점들을 추출
   # Right lane lines
   right_lines_x = []
   right_lines_y = []
   for line in right_lines:
       x1, y1, x2, y2 = line[0]
       right_lines_x.append(x1)
       right_lines_x.append(x2)
       right_lines_y.append(y1)
       right_lines_y.append(y2)

   if len(right_lines_x) > 0:
       right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1) # y = m*x + b
   else:
       right_m, right_b = 1, 1
       draw_right = False

   # Left lane lines
   left_lines_x = []
   left_lines_y = []
   for line in left_lines:
       x1, y1, x2, y2 = line[0]
       left_lines_x.append(x1)
       left_lines_x.append(x2)
       left_lines_y.append(y1)
       left_lines_y.append(y2)

   if len(left_lines_x) > 0:
       left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
   else:
       left_m, left_b = 1, 1
       draw_left = False

   # 왼쪽과 오른쪽의 2개의 점을 찾기, y는 알고 있으므로 x만 찾으면됨
   # y = m*x + b --> x = (y - b)/m
   y1 = img.shape[0]
   y2 = img.shape[0] * (1 - 0.4)
   right_x1 = (y1 - right_b) / right_m
   right_x2 = (y2 - right_b) / right_m
   left_x1 = (y1 - left_b) / left_m
   left_x2 = (y2 - left_b) / left_m

   # 모든 점은 정수형이어야 함(정수형으로 바꾸기)
   y1 = int(y1)
   y2 = int(y2)
   right_x1 = int(right_x1)
   right_x2 = int(right_x2)
   left_x1 = int(left_x1)
   left_x2 = int(left_x2)

   # 차선그리기
   if draw_right:
       cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
   if draw_left:
       cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)



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



   draw_lines(img_src,lines,(0,255,0),thickness=2)




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