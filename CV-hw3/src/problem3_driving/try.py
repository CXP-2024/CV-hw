import os
import sys
import scipy
import numpy as np
import glob
import json
import cv2
from scipy.spatial.transform import Rotation

sys.path.append('../')


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
VIEW_FOV = 100

BB_COLOR = (248, 64, 24)  # the color for drawing bounding boxes


class Agent(object):
	def __init__(self, measure):
		super(Agent, self).__init__()
		self.attr = {}
		for k, v in measure.items():
			if isinstance(v, dict):
				self.attr[k] = Agent(v)
			else:
				self.attr[k] = v

	def __getattr__(self, item):
		return self.attr[str(item)]

	def get_transform(self):
		if 'transform' in self.attr:
			return self.attr['transform']
		else:
			raise Exception

	def __str__(self):
		return self.attr.__str__()


# ==============================================================================
# -- BoundingBoxesTransform ---------------------------------------------------
# ==============================================================================


class BoundingBoxesTransform(object):
	"""
	This is a module responsible for creating 3D bounding boxes and drawing them
	client-side on pygame surface.
	"""

	@staticmethod
	def get_bounding_boxes(nonplayers: list, camera, player):
		"""
		Creates 3D bounding boxes based on nonplayers list and camera.
		:param nonplayers: the list of non-player-agent objects
		:param camera: camera object
		:param player: player object, i.e. the ego car
		:return: the list of coordinates of bounding boxes (each has 8 vertexes), the format is like:
			[matrix([[x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7]]), matrix(...), ...]
		"""
		bounding_boxes = [BoundingBoxesTransform.get_bounding_box(nonplayer, camera, player) for nonplayer in nonplayers]
		return bounding_boxes

	@staticmethod
	def draw_3D_bounding_boxes(image: np.ndarray, bounding_boxes: list):
		"""
		Draws 3D bounding boxes on the input image.
		Do not modify this function! Adjust the format of your bounding boxes to fit this function.
		:param image: image matrix
		:param bounding_boxes: a list of bounding box coordinates
		:return: image array: np.ndarray
		"""
		for bbox in bounding_boxes:
			points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
			# draw lines
			# base
			cv2.line(image, points[0], points[1], color=BB_COLOR, thickness=2)
			cv2.line(image, points[1], points[2], color=BB_COLOR, thickness=2)
			cv2.line(image, points[2], points[3], color=BB_COLOR, thickness=2)
			cv2.line(image, points[3], points[0], color=BB_COLOR, thickness=2)
			# top
			cv2.line(image, points[4], points[5], color=BB_COLOR, thickness=2)
			cv2.line(image, points[5], points[6], color=BB_COLOR, thickness=2)
			cv2.line(image, points[6], points[7], color=BB_COLOR, thickness=2)
			cv2.line(image, points[7], points[4], color=BB_COLOR, thickness=2)
			# base-top
			cv2.line(image, points[0], points[4], color=BB_COLOR, thickness=2)
			cv2.line(image, points[1], points[5], color=BB_COLOR, thickness=2)
			cv2.line(image, points[2], points[6], color=BB_COLOR, thickness=2)
			cv2.line(image, points[3], points[7], color=BB_COLOR, thickness=2)
		return image

	@staticmethod
	def get_bounding_box(nonplayer: Agent, camera: Agent, player: Agent):
		"""
		3D边界框投影核心算法
		坐标系转换流程：
		1. nonplayer local转正
		2. 局部转正体坐标系 -> player 坐标系
		3. player坐标系转正
		4. 平移到相机坐标系并转正
		5. 投影到2D图像坐标系
		"""
		# first compute the real distance between the player and nonplayer
		player_location = np.array([player.transform.location.x, player.transform.location.y, player.transform.location.z])
		nonplayer_location = np.array([nonplayer.transform.location.x, nonplayer.transform.location.y, nonplayer.transform.location.z])
		dis = np.linalg.norm(player_location - nonplayer_location)
		print("\033[1;32mthe distance between player and nonplayer:", dis, "\033[0m")
		# 生成原始边界框顶点（局部坐标系）
		bb_cords = BoundingBoxesTransform._create_bb_points(nonplayer)
		# Try to get transform.location.z first, fallback to extent.z if not available
		try:
			z_box_local_nonplayer = nonplayer.boundingBox.transform.location.z
		except (KeyError, AttributeError):
			z_box_local_nonplayer = 0.0#nonplayer.boundingBox.extent.z# this means the nonplayer is a pedestrian, so we set it to 0.0
			
		try:
			z_box_local_player = player.boundingBox.transform.location.z
		except (KeyError, AttributeError):
			z_box_local_player = 0.0
		print("z_box_local_player: ", z_box_local_player)

		# 获取各坐标系变换参数
		player_transform = BoundingBoxesTransform._complete_transform(player.get_transform())
		nonplayer_transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
		camera_transform = BoundingBoxesTransform._complete_transform(camera.get_transform())

		# ========== 坐标系转换阶段 ==========
		# nonplayer local rotate
		nonplayer_rotate_tran = BoundingBoxesTransform._get_nonplayer_rotate_transform(nonplayer_transform)
		np_rotate = nonplayer_rotate_tran @ bb_cords.T  # 4x8矩阵

		# nonplayer to player
		nonplayer_player_tran = BoundingBoxesTransform._get_player_transform(nonplayer_transform, player_transform, z_box_local_nonplayer, z_box_local_player)
		player_unrotate = nonplayer_player_tran @ np_rotate  # 4x8矩阵

		# player rotate
		player_rotate_tran = BoundingBoxesTransform._get_player_rotate_matrix(player_transform)
		player_rotate = player_rotate_tran @ player_unrotate  # 4x8矩阵

		# player to camera 
		camera_tran = BoundingBoxesTransform._get_camera_matrix(camera_transform, z_box_local_player)
		camera_unrotate = camera_tran @ player_rotate  # 4x8矩阵

		# camera rotate
		camera_rotate_tran = BoundingBoxesTransform._get_view_matrix(camera_transform)
		camera_view = camera_rotate_tran @ camera_unrotate  # 4x8矩阵

		# ========== 投影阶段 ==========
		# Filter out points behind the camera (negative Z)
		if np.all(camera_view[0, :] < 0):
			print("The object is behind the camera")
			print(camera_view[0, :])
			# Return empty box or placeholder if the object is behind camera
			return np.zeros((8, 2))
			
		# Reorder axes for correct projection
		camera_view_ordered = np.zeros((4, 8))
		# X axis in image corresponds to Y in world (right direction)
		camera_view_ordered[0, :] = camera_view[1, :]
		# Y axis in image corresponds to Z in world (up direction)
		camera_view_ordered[1, :] = camera_view[2, :]  
		# Z axis in image (depth) corresponds to X in world (forward direction)
		camera_view_ordered[2, :] = camera_view[0, :]
		camera_view_ordered[3, :] = camera_view[3, :]
		
		# Check if any points are too close to the camera (would cause division by near-zero)
		min_depth = 0.1  # Minimum allowed depth
		if np.any(abs(camera_view_ordered[2, :]) < min_depth):
			# Return empty box or placeholder if the object is too close to camera
			print("The object is too close to the camera")
			print(camera_view_ordered[2, :])
			return np.zeros((8, 2))
		
		# Perspective division - normalize by depth
		points_2d = np.zeros((3, 8))
		points_2d[0, :] = camera_view_ordered[0, :] / camera_view_ordered[2, :]
		points_2d[1, :] = camera_view_ordered[1, :] / camera_view_ordered[2, :]
		points_2d[2, :] = 1.0
		
		# Apply camera calibration matrix
		points_2d = camera.calibration @ points_2d
		points_2d = points_2d[:2, :]  / points_2d[2, :]
		points_2d = points_2d[:2, :]
		
		# Flip Y-axis to match image coordinates (origin at top-left)
		points_2d[1, :] = IMAGE_HEIGHT - points_2d[1, :]
		
		# Filter out any points outside the image bounds with a margin
		margin = 500   # Allow points to be slightly outside the frame
		if (np.any(points_2d[0, :] < -margin) or 
			np.any(points_2d[0, :] > IMAGE_WIDTH + margin) or
			np.any(points_2d[1, :] < -margin) or 
			np.any(points_2d[1, :] > IMAGE_HEIGHT + margin)):
			print("The object is outside the image bounds")
			print(points_2d[0, :])
			print(points_2d[1, :])
			print(camera_view_ordered)
			return np.zeros((8, 2))
		
		# If can go here, print its ok
		print("\033[1;34mThe object is inside the image bounds\033[0m")
			
		return points_2d.T  # 8x2矩阵
		

	@staticmethod
	def _get_nonplayer_rotate_transform(nonplayer_transform):
		"""获取非玩家坐标系变换矩阵"""
		# 非玩家坐标系
		nonplayer_rot = Rotation.from_euler(
			'zyx', [nonplayer_transform.rotation.yaw, # here should be not reverse but the 2nd and 3rd axis maybe reverse or not
					nonplayer_transform.rotation.pitch, 
					-nonplayer_transform.rotation.roll], degrees=True)
		nonplayer_T = np.identity(4)
		nonplayer_T[:3, :3] = nonplayer_rot.as_matrix()
		return nonplayer_T

	@staticmethod
	def _get_player_transform(nonplayer_transform, player_transform, z_box_local_nonplayer, z_box_local_player):
		"""获取非玩家坐标系->玩家坐标系变换矩阵"""
		# 非玩家相对玩家变换
		player_rel = np.identity(4)
		player_rel[:3, 3] = [nonplayer_transform.location.x - player_transform.location.x,
							 nonplayer_transform.location.y - player_transform.location.y,
							 z_box_local_nonplayer - z_box_local_player + nonplayer_transform.location.z - player_transform.location.z]
		return player_rel
	
	@staticmethod
	def _get_player_rotate_matrix(player_transform):
		"""获取玩家坐标系变换矩阵"""
		# 玩家坐标系 here we should rotate the reverse
		player_rot = Rotation.from_euler(
			'zyx', [-player_transform.rotation.yaw, 
					-player_transform.rotation.pitch,
					player_transform.rotation.roll], degrees=True)
		player_T = np.identity(4)
		player_T[:3, :3] = player_rot.as_matrix()
		return player_T

	@staticmethod
	def _get_camera_matrix(camera_transform, z_box_local_player):
		"""获取玩家坐标系->相机坐标系变换矩阵"""
		# camera_transform is just relative to the player
		camera_rel = np.identity(4)
		camera_rel[:3, 3] = [-camera_transform.location.x,
							 -camera_transform.location.y,
							 -camera_transform.location.z + z_box_local_player]
		return camera_rel
	
	@staticmethod
	def _get_view_matrix(camera_transform):
		"""获取相机坐标系变换矩阵"""
		camera_rot = Rotation.from_euler(
			'zyx', [camera_transform.rotation.yaw, 
					camera_transform.rotation.pitch,
					camera_transform.rotation.roll], degrees=True)
		camera_T = np.identity(4)
		camera_T[:3, :3] = camera_rot.as_matrix()
		return camera_T

	@staticmethod
	def _create_bb_points(nonplayer):
		"""
		Returns 3D bounding box for a non-player-agent, relative to the vehicle coordinate system.
		"""

		cords = np.zeros((8, 4))
		extent = nonplayer.boundingBox.extent
		# Scale the extent to make the bounding box more visible
		scale_factor = 1.0  # You can adjust this if needed
		
		cords[0, :] = np.array([extent.x * scale_factor, extent.y * scale_factor, -extent.z * scale_factor, 1])
		cords[1, :] = np.array([-extent.x * scale_factor, extent.y * scale_factor, -extent.z * scale_factor, 1])
		cords[2, :] = np.array([-extent.x * scale_factor, -extent.y * scale_factor, -extent.z * scale_factor, 1])
		cords[3, :] = np.array([extent.x * scale_factor, -extent.y * scale_factor, -extent.z * scale_factor, 1])
		cords[4, :] = np.array([extent.x * scale_factor, extent.y * scale_factor, extent.z * scale_factor, 1])
		cords[5, :] = np.array([-extent.x * scale_factor, extent.y * scale_factor, extent.z * scale_factor, 1])
		cords[6, :] = np.array([-extent.x * scale_factor, -extent.y * scale_factor, extent.z * scale_factor, 1])
		cords[7, :] = np.array([extent.x * scale_factor, -extent.y * scale_factor, extent.z * scale_factor, 1])
		return cords

	@staticmethod
	def _complete_transform(transform):
		"""
		Complete the missing items in transform so avoid raising errors. Maybe useful for you.
		"""
		if 'x' not in transform.location.attr:
			transform.location.x = 0.0
		if 'y' not in transform.location.attr:
			transform.location.y = 0.0
		if 'z' not in transform.location.attr:
			transform.location.z = 0.0
		if 'yaw' not in transform.rotation.attr:
			transform.rotation.yaw = 0.0
		if 'roll' not in transform.rotation.attr:
			transform.rotation.roll = 0.0
		if 'pitch' not in transform.rotation.attr:
			transform.rotation.pitch = 0.0
		return transform


def set_calibration(camera):
	"""
	get the camera calibration matrix
	:param camera: the camera agent object
	:return: camera
	"""
	calibration = np.identity(3)
	
	# Calculate focal length based on FOV
	fov_radians = np.radians(VIEW_FOV)
	focal_length = IMAGE_WIDTH / (2 * np.tan(fov_radians / 2))
	
	# Set calibration matrix
	calibration[0, 0] = focal_length  # fx
	calibration[1, 1] = focal_length  # fy
	calibration[0, 2] = IMAGE_WIDTH / 2  # we'll add the principal point later, The y should be Height - y
	calibration[1, 2] = IMAGE_HEIGHT / 2  # we'll add the principal point later
	
	camera.calibration = calibration
	return camera


def filte_out_near_nonplayer(nonPlayerAgents, playerAgent, threshold=50):
	player_location = np.array([playerAgent.transform.location.x, playerAgent.transform.location.y, playerAgent.transform.location.z])
	near_nonPlayerAgents = []
	for nonplayer in nonPlayerAgents:
		nonplayer_location = np.array([nonplayer.transform.location.x, nonplayer.transform.location.y, nonplayer.transform.location.z])
		dis = np.linalg.norm(player_location - nonplayer_location)
		if dis <= threshold:
			near_nonPlayerAgents.append(nonplayer)
	return near_nonPlayerAgents

'''
In this exercise, you are given three images (src/problem3_driving/data/image *.png) from an au-
tonomous driving simulator and their measurements (src/problem3_driving/data/measurements *.json)
that record the spatial information of all the objects at those moments. For ex-
ample, there are many different measurements in the key playerMeasurements:
the transform represents the location coordinates and rotation pose in the world
coordinate system; the boundingBox is the bounding boxes of the vehicle. And
nonPlayerAgents are many other objects in the simulator, such as other vehi-
cle, pedestrians, traffic lights and etc. Note that, the transform information of
bounding boxes are relative to the agents and that of agents are relative to the
world. For more details, please read the https://carla.readthedocs.io/en/0.8.4/measurements/#player-measurements if necessary.

You are asked to draw the bounding boxes of the non-player-agents in the
corresponding image, like Figure 2. The image shape is (800,600) and the
principle point is just the centre point of the image. The FOV of the camera
is 100. You only need to draw the bounding boxes for two types of agents:
vehicle and pedestrians within 50 metres. Note that, take care of the names
and directions of the axes in Figure 3 because the x-axis is the direction in
which the car is moving.
If you need some help, please refer to the code in https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/client_bounding_boxes.py.
'''

def main():
	img_path_list = glob.glob('src/problem3_driving/data/image_*.png')
	img_path_list.sort()
	measurement_path_list = glob.glob('src/problem3_driving/data/measurements_*.json')
	measurement_path_list.sort()

	for img_path, measurement_path in zip(img_path_list, measurement_path_list):
		# Extract just the filename and then get the index number
		filename = os.path.basename(img_path)
		idx = filename.split('_')[1].split('.')[0]
		image = cv2.imread(img_path)

		with open(measurement_path, 'r') as f:
			measurement = json.load(f)

		nonPlayerAgents = []
		for item in measurement['nonPlayerAgents']:
			if 'vehicle' in item:
				nonPlayerAgents.append(Agent(item['vehicle']))
			elif 'pedestrian' in item:
				nonPlayerAgents.append(Agent(item['pedestrian']))

		playerAgent = Agent(measurement['playerMeasurements'])

		nonPlayerAgents = filte_out_near_nonplayer(nonPlayerAgents, playerAgent)

		camera_player_transform = {'transform': {'location': {'x': 2.0, 'y': 0.0, 'z': 1.4},
												 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
												 'rotation': {"pitch": -15.0, "roll": 0.0, "yaw": 0.0}}}
		camera_agent = Agent(camera_player_transform)
		camera_agent = set_calibration(camera_agent)
		print("length of near nonPlayerAgents: ", len(nonPlayerAgents))

		bounding_boxes = BoundingBoxesTransform.get_bounding_boxes(nonPlayerAgents, camera_agent, playerAgent)
		result = BoundingBoxesTransform.draw_3D_bounding_boxes(image, bounding_boxes)
		print("length of bounding boxes: ", len(bounding_boxes))
		#print("the first bounding box: ", bounding_boxes[0])
		
		# Save with absolute path for reliability
		script_dir = os.path.dirname(os.path.abspath(__file__))
		output_dir = os.path.join(script_dir, 'output')
		os.makedirs(output_dir, exist_ok=True)
		
		output_path = os.path.join(output_dir, f'test_{idx}.png')
		success = cv2.imwrite(output_path, result)
		
		if success:
			print(f"Successfully saved image to {output_path}")
		else:
			print(f"Failed to save image to {output_path}")
		
		# Display the image for immediate feedback
		cv2.imshow(f"Bounding Boxes - Image {idx}", result)
		cv2.waitKey(1000)  # Display for 1 second

	# Wait for key press before closing all windows
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
