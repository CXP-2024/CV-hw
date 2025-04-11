import os
from glob import glob
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def img_depth(disparity: np.ndarray, f: float, baseline: float):
    """
    calculate the image depth from the disparity
    :param disparity: a np.ndarray, shape: (n, m, 1)
    :param f: focal length, type: float
    :param baseline: type: float
    :return: depth matrix, type: np.ndarray, shape: (n, m, 1)
    """
    # The depth is calculated by the formula: depth = f * baseline / disparity
    # f is in mm, baseline is in m, and disparity is in mm
    # here is the code to calculate the depth
    depth = np.zeros(disparity.shape)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    depth[:, :, 0] = f * baseline / (disparity[:, :, 0] + epsilon) # now depth is in m
    # Return raw depth in millimeters
    return depth * 1000

      

def draw_bbox(img, detections):
    """
    draw the 2D bounding boxes on the image
    :param img: the image array, type: np.ndarray
    :param detections: the detection information list, each item is [x_left, y_top, x_right, y_bottom, id, score]
    :return:
    """
    # Loop through each detection and draw the bounding box
    for det in detections:
        # Each detection should be a 6-element array
        x_left, y_top, x_right, y_bottom, obj_id, score = det
        
        # Draw the rectangle on the image
        cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (255, 0, 0), 2)
        
        # Draw the center point of the bounding box
        center_x = int((x_left + x_right) / 2)
        center_y = int((y_top + y_bottom) / 2)
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Draw the text with the ID
        cv2.putText(img, f"ID:{int(obj_id)}", (int(x_left), int(y_top) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    

def coordinate_2d_to_3d(coordinate_q: np.ndarray, f: float, px: float, py: float, depth: np.ndarray):
    """
    The function to convert 2D coordinates in the image plane  to 3D coordinates
    in the world coordinate system with the camera as the origin point
    :param coordinate_q: the pixel coordinate in the image, i.e. q
    :param f: focal length
    :param px: principal point in x axis, type: float
    :param py: principal point in y axis., type: float
    :param depth: depth matrix, type: np.ndarray, shape: (n, m, 1)
    :return: the 3D coordinate in the world
    """
    # The 3D coordinates are calculated by the formula:
    # X = (x - px) * depth / f
    # Y = (y - py) * depth / f
    # Z = depth
    # here f is mm, px, py is mm， and depth is in mm
    # here is the code to calculate the 3D coordinates
    # Get the pixel coordinates
    x = coordinate_q[0]
    y = coordinate_q[1]
    # Get the depth value
    depth_value = depth[y, x, 0]
    # Calculate the 3D coordinates
    X = (x - px) * depth_value / f
    Y = (y - py) * depth_value / f
    Z = depth_value
    # Return the 3D coordinates in meters
    return np.array([X, Y, Z], dtype=np.float32) / 1000  # Convert to meters


def solve_question1():
    """
    Solve the first question: compute the depth map and show the depth map
    """
    # load the disparity map
    # the disparity png lie in the folder: problem2_analyzing/data/detection/   name: num_left_disparity.png total 5 files
    
    # Use glob to find all disparity files
    disparity_files = glob('src/problem2_analyzing/data/detections/*.png')
    if not disparity_files:
        print("No disparity files found. Check the path.")
        return
    
    print(f"Found {len(disparity_files)} disparity files")
    
    # Process each disparity file
    for disparity_file in disparity_files:
        print(f"Processing disparity file: {disparity_file}")
        
        # Load the disparity map
        disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            print(f"Failed to load disparity file: {disparity_file}")
            continue
            
        # convert the disparity to float
        disparity = np.float32(disparity)
        # show the disparity map
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(disparity, cmap='gray')
        plt.title(f'Disparity Map - {os.path.basename(disparity_file)}')
        
        # load the focal length and baseline
        # Get the corresponding calibration file based on the disparity filename
        file_number = os.path.basename(disparity_file).split('_')[0]
        calib_file = f'src/problem2_analyzing/data/calib/{file_number}_allcalib.txt'
        
        # load the txt file
        try:
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                focal_length = float(lines[0].split(':')[1].strip())
                px = float(lines[1].split(':')[1].strip())
                py = float(lines[2].split(':')[1].strip())
                baseline = float(lines[3].split(':')[1].strip())
        except FileNotFoundError:
            print(f"Calibration file not found: {calib_file}")
            print("Falling back to default calibration values...")
            focal_length = 721.537700
            px = 609.559300
            py = 172.854000
            baseline = 0.5327119288
        
        # Compute depth map
        disparity = np.expand_dims(disparity, axis=-1)  # Ensure disparity has shape (n, m, 1)
        depth = img_depth(disparity, focal_length, baseline)
        
        # Display depth map with better visualization
        plt.subplot(1, 2, 2)
        
        # Find reasonable min/max depth values (excluding outliers)
        valid_depths = depth[depth > 0]
        if len(valid_depths) > 0:
            # Use percentiles to exclude extreme values
            min_depth = float(np.percentile(valid_depths, 0.001))  # 0.001th percentile
            max_depth = float(np.percentile(valid_depths, 95))  # 95th percentile
        else:
            min_depth = 0.0
            max_depth = 100.0  # Default max depth in meters
        
        # Use a better colormap for depth visualization (like 'jet', 'plasma', or 'viridis')
        depth_img = plt.imshow(depth[:, :, 0], cmap='plasma', vmin=min_depth, vmax=max_depth)
        plt.title(f'Depth Map - {os.path.basename(disparity_file)}')
        
        # Add colorbar with meter units
        cbar = plt.colorbar(depth_img)
        cbar.set_label('Depth (meters)')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics about the depth
        print(f"Depth statistics for {os.path.basename(disparity_file)}:")
        print(f"  Min depth: {min_depth} meters")
        print(f"  Max depth: {max_depth:.2f} meters")
        print(f"  Mean depth: {np.mean(valid_depths):.2f} meters")
        print(f"  Median depth: {np.median(valid_depths):.2f} meters")
        print(f"  Depth range: {max_depth - min_depth:.2f} meters")
        print(f"  Valid depth count: {len(valid_depths)}")
        
'''
# Under the folder data/detections, there are also the detection results for
# each left image. The results are saved in Matlab (.MAT) format. Use the
# SCIPY.IO.LOADMAT function to load this data. each value in the key dets
# represents a rectangle (called bounding box) around what the detector thinks
# it’s an object. The bounding box is represented with two corner points, the top
# left corner (xlef t, ytop) and the bottom right corner (xright, yright). The value of
# the key dets has the following information: [xlef t, ytop, xright, ybottom, id, score].
# Here SCORE is the confidence of the detection, i.e., it reflects how much a
# detector believes there is an object in that location. The higher the better. The
# variable ID reflects the viewpoint of the detection. You can ignore the SCORE
# and ID for this assignment.
# Visualize the bounding boxes in the images. Then, for each bounding
# box, calculate the 3D coordinate of the diagonal intersection. For convenience,
# use the camera as the origin of the 3D world coordinate system.
'''

def solve_question2():
    """
    Solve the second question: compute the 3D coordinates of the bounding boxes
    """
    # load the detection results
    # the detection mat lie in the folder: problem2_analyzing/data/detection/   name: num_dets.mat total 5 files
    # the png files lie in the folder: problem2_analyzing/data/left   name: num.png total 5 files
    
    # Use glob to find all detection files
    detection_files = glob('src/problem2_analyzing/data/detections/*.mat')
    if not detection_files:
        print("No detection files found. Check the path.")
        return
    print(f"Found {len(detection_files)} detection files")
    # Process each detection file
    for detection_file in detection_files:
        print(f"Processing detection file: {detection_file}")
				
        # Load the detection data
        try:
            detection_data = sio.loadmat(detection_file)
            dets = detection_data['dets']
            print(f"Shape of dets: {dets.shape}")
            
            # Get the actual detections
            all_detections = []
            for i in range(dets.shape[0]):
                if dets[i][0].size > 0:  # Check if it's not an empty array
                    all_detections.extend(dets[i][0])
            
            if not all_detections:
                print(f"No valid detections in file: {detection_file}")
                continue
                
            print(f"Found {len(all_detections)} valid detections")
        except Exception as e:
            print(f"Failed to load detection file: {detection_file}, Error: {e}")
            continue
        
        # Load the corresponding image
        image_file = os.path.join('src/problem2_analyzing/data/left', os.path.basename(detection_file).replace('_dets.mat', '.jpg'))
        if not os.path.exists(image_file):
            print(f"Image file not found: {image_file}")
            continue
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image file: {image_file}")
            continue
        # Convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Draw the bounding boxes on the image
        draw_bbox(img, all_detections)
        # Show the image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f'Bounding Boxes - {os.path.basename(detection_file)}')
        plt.show()
        
        # Load the corresponding disparity map
        disparity_file = os.path.join('src/problem2_analyzing/data/detections', os.path.basename(detection_file).replace('_dets.mat', '_left_disparity.png'))
        if not os.path.exists(disparity_file):
            print(f"Disparity file not found: {disparity_file}")
            continue
        disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            print(f"Failed to load disparity file: {disparity_file}")
            continue
        # Convert the disparity to float
        disparity = np.float32(disparity)
        # Load the calibration data
        # Get the corresponding calibration file based on the disparity filename
        file_number = os.path.basename(detection_file).split('_')[0]
        calib_file = f'src/problem2_analyzing/data/calib/{file_number}_allcalib.txt'
        
        # load the txt file
        try:
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                focal_length = float(lines[0].split(':')[1].strip())
                px = float(lines[1].split(':')[1].strip())
                py = float(lines[2].split(':')[1].strip())
                baseline = float(lines[3].split(':')[1].strip())
        except FileNotFoundError:
            print(f"Calibration file not found: {calib_file}")
            print("Falling back to default calibration values...")
            focal_length = 721.537700
            px = 609.559300
            py = 172.854000
            baseline = 0.5327119288
            
        # Compute depth map
        disparity = np.expand_dims(disparity, axis=-1)
        depth = img_depth(disparity, focal_length, baseline)
        # Convert the depth type to float32 while preserving array structure
        depth = depth.astype(np.float32)
        # Calculate the 3D coordinates for each detection
        for det in all_detections:
            x_left, y_top, x_right, y_bottom, obj_id, score = det
            # Calculate the center of the bounding box
            center_x = int((x_left + x_right) / 2)
            center_y = int((y_top + y_bottom) / 2)
            # Get the 3D coordinates
            coordinate_q = np.array([center_x, center_y])
            coordinate_3d = coordinate_2d_to_3d(coordinate_q, focal_length, px, py, depth)
            print(f"3D Coordinates for detection ID {int(obj_id)}: {coordinate_3d}")

  
def main():
    ############    load the data, implement the above three functions and solve the two questions   #######
    # Call the function to solve question 1
    solve_question1()
    #solve_question2()
    
    # Question 2 code would go here
    # ...


if __name__ == '__main__':
    main()
