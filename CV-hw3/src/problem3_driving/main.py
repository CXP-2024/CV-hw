import os
import sys
import scipy
import numpy as np
import glob
import json
import cv2

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
        Returns 3D bounding box for a vehicle based on camera view.
        :param nonplayer: the non-player object
        :param camera: the camera object
        :param player: the player object, i.e. the ego vehicle
        :return: the 2D coordinates of the bounding box vertexes
        """
        bb_cords = BoundingBoxesTransform._create_bb_points(nonplayer)  # the 3D coordinates of the bounding box
        
        # Complete transformation for player and nonplayer
        player_transform = BoundingBoxesTransform._complete_transform(player.get_transform())
        nonplayer_transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
        camera_transform = BoundingBoxesTransform._complete_transform(camera.get_transform())
        
        # Create transformation matrices
        # Non-player to world transformation
        nonplayer_world_transform = np.identity(4)
        # Rotation matrix - convert from Euler angles (roll, pitch, yaw) to rotation matrix
        nonplayer_roll = np.radians(nonplayer_transform.rotation.roll)
        nonplayer_pitch = np.radians(nonplayer_transform.rotation.pitch)
        nonplayer_yaw = np.radians(nonplayer_transform.rotation.yaw)
        
        # Create rotation matrix using Euler angles (roll, pitch, yaw)
        cos_roll, sin_roll = np.cos(nonplayer_roll), np.sin(nonplayer_roll)
        cos_pitch, sin_pitch = np.cos(nonplayer_pitch), np.sin(nonplayer_pitch)
        cos_yaw, sin_yaw = np.cos(nonplayer_yaw), np.sin(nonplayer_yaw)
        
        # Rotation matrices
        roll_matrix = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        pitch_matrix = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        yaw_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix
        
        # Set rotation part of the transformation matrix
        nonplayer_world_transform[:3, :3] = rotation_matrix
        
        # Set translation part
        nonplayer_world_transform[0, 3] = nonplayer_transform.location.x
        nonplayer_world_transform[1, 3] = nonplayer_transform.location.y
        nonplayer_world_transform[2, 3] = nonplayer_transform.location.z
        
        # World to camera transformation
        world_camera_transform = np.identity(4)
        
        # Camera rotation matrix
        camera_roll = np.radians(camera_transform.rotation.roll)
        camera_pitch = np.radians(camera_transform.rotation.pitch)
        camera_yaw = np.radians(camera_transform.rotation.yaw)
        
        # Create camera rotation matrices
        c_roll_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(camera_roll), -np.sin(camera_roll)],
            [0, np.sin(camera_roll), np.cos(camera_roll)]
        ])
        
        c_pitch_matrix = np.array([
            [np.cos(camera_pitch), 0, np.sin(camera_pitch)],
            [0, 1, 0],
            [-np.sin(camera_pitch), 0, np.cos(camera_pitch)]
        ])
        
        c_yaw_matrix = np.array([
            [np.cos(camera_yaw), -np.sin(camera_yaw), 0],
            [np.sin(camera_yaw), np.cos(camera_yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined camera rotation matrix
        camera_rotation_matrix = c_yaw_matrix @ c_pitch_matrix @ c_roll_matrix
        
        # Player position/rotation in world coordinates
        player_transform_matrix = np.identity(4)
        
        # Player rotation
        player_roll = np.radians(player_transform.rotation.roll)
        player_pitch = np.radians(player_transform.rotation.pitch)
        player_yaw = np.radians(player_transform.rotation.yaw)
        
        p_roll_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(player_roll), -np.sin(player_roll)],
            [0, np.sin(player_roll), np.cos(player_roll)]
        ])
        
        p_pitch_matrix = np.array([
            [np.cos(player_pitch), 0, np.sin(player_pitch)],
            [0, 1, 0],
            [-np.sin(player_pitch), 0, np.cos(player_pitch)]
        ])
        
        p_yaw_matrix = np.array([
            [np.cos(player_yaw), -np.sin(player_yaw), 0],
            [np.sin(player_yaw), np.cos(player_yaw), 0],
            [0, 0, 1]
        ])
        
        player_rotation_matrix = p_yaw_matrix @ p_pitch_matrix @ p_roll_matrix
        player_transform_matrix[:3, :3] = player_rotation_matrix
        player_transform_matrix[0, 3] = player_transform.location.x
        player_transform_matrix[1, 3] = player_transform.location.y
        player_transform_matrix[2, 3] = player_transform.location.z
        
        # Camera position relative to player
        camera_player_transform = np.identity(4)
        camera_player_transform[:3, :3] = camera_rotation_matrix
        camera_player_transform[0, 3] = camera_transform.location.x
        camera_player_transform[1, 3] = camera_transform.location.y
        camera_player_transform[2, 3] = camera_transform.location.z
        
        # Total transformation: local to world to camera
        camera_to_world = player_transform_matrix @ camera_player_transform
        world_to_camera = np.linalg.inv(camera_to_world)
        
        # Apply transformations
        local_to_world = nonplayer_world_transform @ bb_cords.T
        world_to_cam = world_to_camera @ local_to_world
        
        # Project 3D to 2D
        points_3d = world_to_cam[:3, :]
        
        # Only keep points in front of the camera
        points_in_front = points_3d[2, :] > 0
        
        if not np.any(points_in_front):
            return np.zeros((8, 2))
            
        # Normalize by Z coordinate
        points_2d = np.zeros((3, points_3d.shape[1]))
        points_2d[0, :] = points_3d[0, :] / points_3d[2, :]
        points_2d[1, :] = points_3d[1, :] / points_3d[2, :]
        points_2d[2, :] = 1
        
        # Apply camera calibration
        image_points = camera.calibration @ points_2d
        
        # Return as (8, 2) numpy array as expected by draw_3D_bounding_boxes
        return np.array([[image_points[0, i], image_points[1, i]] for i in range(8)])

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
