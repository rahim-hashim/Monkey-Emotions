import os
import sys
import cv2

def draw_point(frame, points, colors):
	'''
	Draws a blue circle at the center of each point.

	Parameters:
		frame (numpy.ndarray): The frame to draw the circle on.
		points (list): A list of x/y positions to draw the circle at.

	Returns:
		frame (numpy.ndarray): The frame with the circle drawn.
	'''
	height, width = frame.shape
	for pix, point in enumerate(points):
		# calculate the center of the point
		center = (point[0], point[1])
		radius = 10
		color = colors[pix]
		# fill the circle
		thickness = -1
		# draw the filled circle
		frame = cv2.circle(frame, center, radius, color, thickness)
	return frame

def canny_detection(video_path):
	'''
	Applies Canny edge detection to a video file and return 
	a new video file with the edges detected.

	Parameters:
		video_path (str): The path to the video file.

	Returns:
		new_video_path (str): The path to the new video file with the edges detected.
	'''
	cap = cv2.VideoCapture(video_path)
	# get the frame rate of the video
	fps = cap.get(cv2.CAP_PROP_FPS)
	new_video_path = os.path.join(os.path.dirname(video_path), 'canny_' + os.path.basename(video_path))
	new_video_frames = []
	points = [(100, 100), (200, 200), (300, 300)]
	colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
				break
		# apply Canny edge detection
		edges = cv2.Canny(frame, 75, 100)
		# draw colored points on the canny edge frame
		frame = draw_point(edges, points, colors)
		# add the edges to the frame
		new_video_frames.append(frame)
	cap.release()
	print(f'Number of frames: {len(new_video_frames)}')
	# write the new video file with fps, width, and height of the original video
	height, width = new_video_frames[0].shape
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))
	for frame in new_video_frames:
		out.write(frame)
	out.release()
	return new_video_path

if __name__ == '__main__':
	video_path = sys.argv[1]
	canny_detection(video_path)