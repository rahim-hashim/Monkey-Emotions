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
	height, width, _ = frame.shape
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

def canny_detection(trial, 
										epochs, 
										video_path,
										slowdown=1):
	'''
	Applies Canny edge detection to a video file and return 
	a new video file with the edges detected.

	Parameters:
		trial (pd.DataFrame): The DataFrame containing the trial data.
		epochs (dict): A dictionary containing the start and end frames for each epoch.
		video_path (str): The path to the video file.
		slowdown (int): To slow down the video by a factor of `slowdown` (default=1).

	Returns:
		new_video_path (str): The path to the new video file with the edges detected.
	'''
	# eye positions
	eye_positions = list(zip(trial['eye_x'].tolist()[0],
					   				 trial['eye_y'].tolist()[0]))
	print(f'  Eye Positions: {len(eye_positions)}')
	# check if video path exists
	if not os.path.exists(video_path):
		raise FileNotFoundError(f'Video file not found: {video_path}')
	print(f'  Video: {video_path}')
	cap = cv2.VideoCapture(video_path)
	# get the frame rate of the video
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(f'    Original FPS: {fps}')
	width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	new_video_path = os.path.join(os.path.dirname(video_path), 
				'canny_' + os.path.basename(video_path.replace('.mp4', '.avi')))
	# create a VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	if slowdown != 1:
		print(f'       Slowing Video by: {slowdown}x')
		print(f'    Final FPS: {round(fps/slowdown)}')
		new_video_path = new_video_path.replace('.avi', f'_slow{slowdown}.avi')
	print(f'    Width x Height: {width}x{height}')
	fps = round(fps/slowdown)
	out = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))
	points = [(100, 100), (200, 200), (300, 300)]
	colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
	new_frames = []
	cam_frames = trial['cam_frames'].tolist()[0]
	print(f'    Num Frames (cam_frames): {len(cam_frames)}')
	# print number of frames in video
	print(f'    Num Frames (video): {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
	print(f'    Frames: {cam_frames[0]}-{cam_frames[-1]}')
	frame_num = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		edges = cv2.cvtColor(cv2.Canny(frame,50,100), cv2.COLOR_GRAY2BGR)
		# convert to RGB
		edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
		# draw the point
		# edges = draw_point(edges, points, colors)
		new_frames.append(edges)
	# write the new video file
	for frame in new_frames:
		out.write(frame)
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	# write the new video file with fps, width, and height of the original video
	return new_video_path

if __name__ == '__main__':
	video_path = sys.argv[1]
	canny_detection(video_path)