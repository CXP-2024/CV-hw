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
		1. 物体局部坐标系 -> 世界坐标系
		2. 世界坐标系 -> 相机坐标系
		3. 投影到2D图像平面
		"""
		# 生成原始边界框顶点（局部坐标系）
		bb_cords = BoundingBoxesTransform._create_bb_points(nonplayer)
		

		# 获取各坐标系变换参数
		player_transform = BoundingBoxesTransform._complete_transform(player.get_transform())
		nonplayer_transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
		camera_transform = BoundingBoxesTransform._complete_transform(camera.get_transform())

		# ========== 坐标系转换阶段 ==========
		# 阶段1：局部 -> 世界坐标系
		nonplayer_world = BoundingBoxesTransform._get_world_transform(nonplayer_transform)
		local_to_world = nonplayer_world @ bb_cords.T  # 4x8矩阵

		# 阶段2：世界 -> 相机坐标系
		world_to_cam = BoundingBoxesTransform._get_view_matrix(player_transform, camera_transform)
		world_to_cam_points = world_to_cam @ local_to_world  # 4x8矩阵

		# ========== 投影阶段 ==========
		# 阶段3：3D->2D投影（透视除法）
		points_3d = world_to_cam_points[:3, :]
		points_2d = camera.calibration @ (points_3d / points_3d[2:3, :])  # 归一化并应用内参

		# ========== 调试输出：关键转换结果 ==========
		print("[DEBUG] World Coordinates:")
		print(local_to_world[:3, :4].T)
		
		print("[DEBUG] Camera Coordinates:")
		print(world_to_cam_points[:3, :4].T)
		
		print("[DEBUG] Projected 2D Points:")
		print(points_2d[:2, :].T)

		# 格式转换并返回（8x2矩阵）
		return np.array([[points_2d[0, i], points_2d[1, i]] for i in range(8)])

	@staticmethod
	def _get_world_transform(transform):
		"""构建物体世界坐标系变换矩阵（含旋转和平移）"""
		# 欧拉角转弧度
		roll = np.radians(transform.rotation.roll)
		pitch = np.radians(transform.rotation.pitch)
		yaw = np.radians(transform.rotation.yaw)
		
		# 旋转矩阵（ZYX顺序：yaw -> pitch -> roll）
		R = (
			Rotation.from_euler('z', yaw, degrees=False).as_matrix() @ 
			Rotation.from_euler('y', pitch, degrees=False).as_matrix() @ 
			Rotation.from_euler('x', roll, degrees=False).as_matrix()
		)
		
		# 齐次变换矩阵
		T = np.identity(4)
		T[:3, :3] = R
		T[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]
		return T

	@staticmethod
	def _get_view_matrix(player_transform, camera_transform):
		"""构建世界->相机视图矩阵"""
		# 相机相对玩家变换
		cam_rel = np.identity(4)
		cam_rel[:3, 3] = [camera_transform.location.x, 
							camera_transform.location.y,
							camera_transform.location.z]
		
		# 玩家世界坐标系
		player_rot = Rotation.from_euler(
			'zyx', [player_transform.rotation.yaw, 
					player_transform.rotation.pitch,
					player_transform.rotation.roll], degrees=True)
		player_T = np.identity(4)
		player_T[:3, :3] = player_rot.as_matrix()
		player_T[:3, 3] = [player_transform.location.x,
							player_transform.location.y,
							player_transform.location.z]
		
		# 组合变换矩阵
		view_matrix = np.linalg.inv(player_T @ cam_rel)
		return view_matrix

	@staticmethod
	def _create_bb_points(nonplayer):
		"""
		Returns 3D bounding box for a non-player-agent, relative to the vehicle coordinate system.
		"""

		cords = np.zeros((8, 4))
		extent = nonplayer.boundingBox.extent
		cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
		cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
		cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
		cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
		cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
		cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
		cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
		cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
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
	calibration[0, 2] = IMAGE_WIDTH / 2.0  # cx - principal point x
	calibration[1, 2] = IMAGE_HEIGHT / 2.0  # cy - principal point y
	
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
		print("the first bounding box: ", bounding_boxes[0])
		
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
