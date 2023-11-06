import cv2
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# cvzone
# import cvzone
# from cvzone.PlotModule import LivePlot
# from cvzone.FaceMeshModule import FaceMeshDetector
# Custom classes
from classes import FaceLandmarks

# # cvzone live plotting
# plot_y = LivePlot(320,240,[30,50])

def calculate_area(rectangle):
  '''Calculates the area of a rectangle'''
  x, y, w, h = rectangle
  return w * h

def is_overlap(rectangle1, rectangle2):
  '''Determines if two rectanges are overlapping.'''
  x1, y1, w1, h1 = rectangle1
  x2, y2, w2, h2 = rectangle2
  if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
    return True
  else:
    return False

def find_non_overlapping_rectangles(rectangle_list):
  final_rectangle_list = rectangle_list.copy()
  rectangle_list_combinations = itertools.combinations(rectangle_list, 2)
  for rectangle_1, rectangle_2 in rectangle_list_combinations:
      if is_overlap(rectangle_1, rectangle_2):
        # only remove rectangles that are overlapping and larger
        if calculate_area(rectangle_1) < calculate_area(rectangle_2):
          final_rectangle_list = \
            [rectangle for rectangle in rectangle_list if list(rectangle) != list(rectangle_2)]
        else:
          [rectangle for rectangle in rectangle_list if list(rectangle) != list(rectangle_1)]
  return np.array(final_rectangle_list)


# def frame_eye_capture(image, trial_iter):
#   # Initialize FaceLandmarks object
#   face_landmarks = FaceLandmarks.FaceLandmarks()
#   # cvzone face detection (utilizing mediapipe)
#   detector = FaceMeshDetector(maxFaces=1)
#   BLINK_WINDOW = 10

#   # Initialize variables
#   frame_number = 0
#   frame_numbers = []
#   ratio_eye_dist_list = []
#   missing_eye_frames = []
#   running_avg = []
#   blink_counter = 0
#   counter = 0
#   left_eye_center_dict = defaultdict(int)
#   right_eye_center_dict = defaultdict(int)

#   # cv2 haarcascade
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   # Detect the face(s)
#   faces = face_landmarks.face_cascade.detectMultiScale(gray, 1.1, 4)
#   # Draw the rectangle around each face
#   for (x, y, w, h) in faces:
#     # cv2.rectangle(image_final, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = image[y:y+h, x:x+w]
#     # Detects eyes of different sizes in the input image
#     eyes = face_landmarks.eye_cascade.detectMultiScale(roi_gray)
#     # Overlap between eye bounding boxes
#     if len(eyes) < 2:
#       eyes_nodup = np.array(eyes)
#     if len(eyes) == 2:
#       eyes_nodup = np.array(eyes)
#       ex1, ex2 = eyes_nodup[0][0], eyes_nodup[1][0]
#       (left_eye, right_eye) = \
#         (eyes_nodup[0], eyes_nodup[1]) if ex1 < ex2 else (eyes_nodup[1], eyes_nodup[0])
#     if len(eyes) > 2:
#       eyes_nodup = find_non_overlapping_rectangles(eyes)
#       break
#     for (ex, ey, ew, eh) in eyes_nodup:
#       cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
#       left_eye_center_dict[frame_number] = ex+(ew/2)
#       right_eye_center_dict[frame_number] = ey+(eh/2)
#     # only one face
#     break

#   ## mediapipe face_mesh
#   height, width, _ = image.shape
#   rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#   # Detect the face
#   image, faces = detector.findFaceMesh(image, draw=False)
#   left_down, left_up = (0,0), (0,0)
#   left_right, left_left = (0,0), (0,0)
#   if faces:
#     face = faces[0]
#     # all facial landmarks
#     # for i in range(0, 468):
#     all_eye_ids = face_landmarks.eyes
#     for id in all_eye_ids:
#       pt_xy = face[id]
#       # cv2.circle(image, pt_xy, 2, (255,0,255), cv2.FILLED) # turn on if you want to see faces
#       if id == face_landmarks.left_eye_bottom:
#         left_down = pt_xy
#       if id == face_landmarks.left_eye_top:
#         left_up = pt_xy
#       if id == face_landmarks.left_eye_right:
#         left_right = pt_xy
#       if id == face_landmarks.left_eye_left:
#         left_left = pt_xy
#     # Calculate distance between left/right and up/down portions of yee
#     length_vert = math.dist(left_down, left_up)
#     length_horz = math.dist(left_right, left_left)
#     cv2.line(image, left_right, left_left, (0,200,0)) # turn on if you want to see faces
#     cv2.line(image, left_up, left_down, (0,200,0))    # turn on if you want to see faces
#     ratio_eye_dist = int((length_vert/length_horz)*100)
#   else:
#     ratio_eye_dist = np.nan
#     missing_eye_frames.append(frame_number)
#     # image_plot = plot_y.update(ratio_eye_dist)
#   # blink counter
#   running_avg.append(ratio_eye_dist)
#   if len(running_avg) > BLINK_WINDOW:
#     running_avg.pop(0)
#   nan_count = np.count_nonzero(np.isnan(running_avg))
#   nonnan_count = np.count_nonzero(~np.isnan(running_avg))
#   ratio_avg = round(np.nansum(running_avg)/nonnan_count, 1)
#   if ratio_avg < 35 and counter == 0:
#     blink_counter += 1
#     counter = 1
#   # blink frame lag = 10 frames
#   if counter != 0:
#     counter += 1
#     if counter > 10:
#       counter = 0
#   return image