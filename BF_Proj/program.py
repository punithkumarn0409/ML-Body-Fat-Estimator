import mediapipe as mp
import cv2
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import sys
import numpy as np
import math
import pandas as pb
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import uuid

height_in_cm = float(183)
#img_path = "InputImage.jpg"

def find_bodyfat(person_height_cm,image_path):
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic

    # Load an image

    image = cv2.resize(cv2.imread(image_path), (750,750))

    cv2.imwrite('resized_image.jpg', image)

    img_height, img_width, _ = image.shape

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Process the image with MediaPipe
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #     results = holistic.process(image_rgb)

    results = holistic.process(image_rgb)

    point23 = int(results.pose_landmarks.landmark[23].x*img_width), int(results.pose_landmarks.landmark[23].y*img_height)


    point24 = int(results.pose_landmarks.landmark[24].x*img_width), int(results.pose_landmarks.landmark[24].y*img_height)


    point12 = int(results.pose_landmarks.landmark[12].x*img_width), int(results.pose_landmarks.landmark[12].y*img_height)


    point11 = int(results.pose_landmarks.landmark[11].x*img_width), int(results.pose_landmarks.landmark[11].y*img_height)


    point16 = int(results.pose_landmarks.landmark[16].x*img_width), int(results.pose_landmarks.landmark[16].y*img_height)

    point15 = int(results.pose_landmarks.landmark[15].x*img_width), int(results.pose_landmarks.landmark[15].y*img_height)

    point28 = int(results.pose_landmarks.landmark[28].x*img_width), int(results.pose_landmarks.landmark[28].y*img_height)

    landmarkedimage = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(landmarkedimage, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    #cv2.imwrite('LandmarkedImage.jpg', landmarkedimage)

    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")
    target_classes = ins.select_target_classes(person = True)
    results, output = ins.segmentImage("resized_image.jpg", segment_target_classes=target_classes, show_bboxes=True, mask_points_values=True, extract_segmented_objects=True)

    max_sub_mask = None
    max_sub_mask = results['masks'][0]

    range_threshold = 5  # Define your range threshold here
    coordinates24 = []
    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
        
            if abs(y - point24[1]) <= range_threshold:  # Check if y is within the range threshold
                coordinates24.append((x, y))

    waist_left_coordinate = None
    prev_x = None
    for x, y in coordinates24:
        if prev_x is not None and abs(x -prev_x) > 100:
              waist_left_coordinate = x, y
              break
        else:
            prev_x = x



    #if waist_left_coordinate is not None:       
    
    if waist_left_coordinate is None:
        min_x_for_y = {}
        range_threshold = 50  # Define your range threshold here

        for sub_array in max_sub_mask:
            for coord in sub_array:
                x, y = coord
            
                if abs(y - point24[1]) <= range_threshold:  # Check if y is within the range threshold
                    if y in min_x_for_y:
                        min_x_for_y[y] = min(min_x_for_y[y], x)
                    else:
                        min_x_for_y[y] = x

        # Check if min_x_for_y is empty
        if not min_x_for_y:
            print("No points found within the range of point23[1].")

        # Find the minimum x-coordinate and corresponding y-coordinate
        waist_left_coordinate = min(min_x_for_y.items(), key=lambda x: x[1])

        # If you only want the x-coordinate
        waist_left_coordinate = waist_left_coordinate[1], point24[1]
    
        # print("waist left coordinates",waist_left_coordinate)




    range_threshold = 5  # Define your range threshold here
    coordinates23 = []
    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
        
            if abs(y - point23[1]) <= range_threshold:  # Check if y is within the range threshold
                coordinates23.append((x, y))


    prev_x = None
    waist_right_coordinate = None
    for x, y in reversed(coordinates23):
        if prev_x is not None and abs(x - prev_x) > 100:
              waist_right_coordinate = x, y
              break
        else:
            prev_x = x


    if waist_right_coordinate is None:
        max_x_for_y = {}
        range_threshold = 50  # Define your range threshold here

        for sub_array in max_sub_mask:
            for coord in sub_array:
                x, y = coord
            
                if abs(y - point23[1]) <= range_threshold:  # Check if y is within the range threshold
                    if y in max_x_for_y:
                        max_x_for_y[y] = max(max_x_for_y[y], x)
                    else:
                        max_x_for_y[y] = x

        # Check if max_x_for_y is empty
        if not max_x_for_y:
            print("No points found within the range of point23[1].")

        # Find the maximum x-coordinate and corresponding y-coordinate
        waist_right_coordinate = max(max_x_for_y.items(), key=lambda x: x[1])

        waist_right_coordinate = waist_right_coordinate[1], point23[1]





    range_threshold = 5
    shoulder_left_coordinate =None
    temp_shoulder_coordinate = point12

    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
            if abs(x - point12[0])<=range_threshold and  y < temp_shoulder_coordinate[1]:
                temp_shoulder_coordinate = x, y

    if temp_shoulder_coordinate is not None:
        x_top, y_top = temp_shoulder_coordinate
        for sub_array in max_sub_mask:
            for coord in sub_array:
                x, y = coord
                if abs(y -y_top)<=range_threshold and x < x_top and (shoulder_left_coordinate is None or x < shoulder_left_coordinate[0]):
                    shoulder_left_coordinate = x, y


    if shoulder_left_coordinate is None:
        shoulder_left_coordinate = temp_shoulder_coordinate
    





    range_threshold = 5
    shoulder_right_coordinate =None
    temp_shoulder_coordinate = point11

    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
            if abs(x - point11[0])<=range_threshold and  y < temp_shoulder_coordinate[1]:
                temp_shoulder_coordinate = x, y

    if temp_shoulder_coordinate is not None:
        x_top, y_top = temp_shoulder_coordinate
        for sub_array in max_sub_mask:
            for coord in sub_array:
                x, y = coord
                if abs(y -y_top)<=range_threshold and x > x_top and (shoulder_right_coordinate is None or x > shoulder_right_coordinate[0]):
                    shoulder_right_coordinate = x, y


    if shoulder_right_coordinate is None:
        shoulder_right_coordinate = temp_shoulder_coordinate
    





    y_min = float('inf')
    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
            if y < y_min:
                y_min = y
                top_coordinate = (x, y)


    y_min = -float('inf')
    for sub_array in max_sub_mask:
        for coord in sub_array:
            x, y = coord
            if y > y_min:
                y_min = y
                bottom_coordinate = (top_coordinate[0], y)
 









    def euclidean_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    person_height_pixels = euclidean_distance(top_coordinate, bottom_coordinate)
    pixel_per_cm = float(person_height_pixels) / float(person_height_cm)

    distance_shoulder = euclidean_distance(shoulder_left_coordinate, shoulder_right_coordinate)
    distance_shoulder = distance_shoulder / pixel_per_cm

    distance_waist = euclidean_distance(waist_left_coordinate, waist_right_coordinate)
    distance_waist = distance_waist / pixel_per_cm

    distance_arms = euclidean_distance(point16, shoulder_left_coordinate)
    distance_arms = distance_arms / pixel_per_cm


    distance_arms2 = euclidean_distance(point15, shoulder_right_coordinate)
    distance_arms2 = distance_arms2 / pixel_per_cm

    distance_legs = euclidean_distance(point24, point28)
    distance_legs = distance_legs / pixel_per_cm



    waist = distance_waist

    p_waist = (2*3.14)*math.sqrt((pow(0.6*waist,2)+pow(waist,2))/2)



    f_waist = p_waist/(person_height_cm)


    df = pb.read_csv('DS.csv')

    total_bf = 0
    total_wt = 0

    for index,row in df.iterrows():
      hip_sc = 10/abs(f_waist-row['hip'])
      total_bf+= hip_sc*row['bf']
      total_wt+=hip_sc
  
    return str(total_bf/total_wt)[:5]

#print(find_bodyfat(height_in_cm,img_path))

app = Flask(__name__)

# Enable CORS for all routes or you can customize the origins
CORS(app)  # This allows cross-origin requests from any domain

@app.route('/bfestimate', methods=['POST'])
def infer_image():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify(error="Please upload the image"), 400

    file = request.files.get('file')

    # Check if height is provided
    usrheight = request.form.get('height')
    if not usrheight:
        return jsonify(error="Please provide height"), 400

    try:
        usrheight = float(usrheight)
    except ValueError:
        return jsonify(error="Height must be a valid number"), 400

    # Generate a unique file name to prevent overwriting
    file_extension = file.filename.split('.')[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    img_path = os.path.join("./images", file_name)

    # Save the file
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    file.save(img_path)

    # Process the image and calculate body fat
    try:
        res = find_bodyfat(usrheight, img_path)
    except Exception as e:
        return jsonify(error=f"Error in processing body fat: {str(e)}"), 500

    # Return the result as JSON
    return jsonify(result=res)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')


