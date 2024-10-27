import logging
import cv2 as cv
import tensorflow as tf
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import numpy as np
import joblib

from utils import CvFpsCalc
from model import KeyPointClassifier

MEDIAPIPE_HEIGHT = 256
MEDIAPIPE_WIDTH = 256
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SCORE_THRESHOLD = 0.5

def main():

    use_brect = True
    mode = 0

    # 0. Initialize
    MODEL_PATH_HAND = "C:\AI-buzz\project_main\model\mediapipe_hand-mediapipehanddetector.tflite"
    MODEL_PATH_LANDMARK = "C:\AI-buzz\project_main\model\mediapipe_hand-mediapipehandlandmarkdetector.tflite"
    MODEL_DT = "C:\AI-buzz\project_main\DT_model.pkl"
    #LABEL_PATH = utils.get_label_path()

    keypoint_classifier = KeyPointClassifier()
    Dtree = joblib.load(MODEL_DT)

    with open('model/keypoint_classifier/keypoint_classifier_label_akshay.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        logging.error("Failed to open camera, exiting")
        exit(1)

    frame_width = FRAME_WIDTH
    frame_height = FRAME_HEIGHT

     # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Load the TFLite model and allocate tensors.
    interpreter_hand = tf.lite.Interpreter(model_path=MODEL_PATH_HAND)
    interpreter_landmark = tf.lite.Interpreter(model_path=MODEL_PATH_LANDMARK)
    interpreter_hand.allocate_tensors()
    interpreter_landmark.allocate_tensors()

    caption = ""

    #class_names = utils.get_yolov8_classes()

    # 1. Capture image from camera
    while True:

        fps = cvFpsCalc.get()

        key = cv.waitKey(100)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        success, frame = camera.read()

        # Determining frame width and frame height 
        # frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))               
        # print(frame_width, frame_height)

        if not success:
            break
        image = cv.flip(frame, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # 2. Pre-process image
        #input_tensor, resized_frame = tf_utils.preprocess_for_yolov8(frame)
        input_tensor, resized_frame = preprocess_for_mediapipe(debug_image)

        # 3. Execute model and get inference results

        # Get input and output tensors for hand detection.
        input_details = interpreter_hand.get_input_details()

        #print(input_details)
        output_details = interpreter_hand.get_output_details()
        #print(output_details)

        # Run the model on input tensor.
        interpreter_hand.set_tensor(input_details[0]["index"], input_tensor)
        interpreter_hand.invoke()

        box_coord = interpreter_hand.get_tensor(output_details[0]["index"]).reshape(2944, 18)
        box_scores = interpreter_hand.get_tensor(output_details[1]["index"]).reshape(2944)

        # Get input and output tensors for landmark detection detection.
        input_details = interpreter_landmark.get_input_details()
        output_details = interpreter_landmark.get_output_details()

        #print(output_details)

        # Run the model on input tensor.
        interpreter_landmark.set_tensor(input_details[0]["index"], input_tensor)
        interpreter_landmark.invoke()

        score = interpreter_landmark.get_tensor(output_details[0]["index"])  # .reshape(2944, 18)
        lr = interpreter_landmark.get_tensor(output_details[1]["index"]) # .reshape(2944)
        hand_landmarks = interpreter_landmark.get_tensor(output_details[2]["index"]).reshape(21, 3)

        # for hand_landmark in hand_landmarks_raw:
        #     landmark = {'x': hand_landmark[0], 'y': hand_landmark[1], 'z': hand_landmark[2]}
        #     hand_landmarks.append(landmark)

        # print(hand_landmarks)
        # continue
        #print(box_scores)
        #print(box_coord)
        #print(scores, lr)  # whether hand is present or not
        #print(lr)     # whether hand is right or not

        # x = [[0 for i in range(2)] for j in range(21)]


        # for index, hand_landmark in enumerate(hand_landmarks):
        #     x[index][0] = hand_landmark[0]
        #     x[index][1] = hand_landmark[1]

        # print(x)

        if score > SCORE_THRESHOLD:
            #for hand_landmarks, handedness in zip(results.multi_hand_landmarks):
                
            #print(results.multi_hand_landmarks)
            #print(handedness)
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list_new(debug_image, hand_landmarks)

            #print(landmark_list)
            #print(len(landmark_list), len(landmark_list[0]))

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark_akshay(
                landmark_list)
            
            #print(len(pre_processed_landmark_list))
            # pre_processed_point_history_list = pre_process_point_history(
            #     debug_image, point_history)
            # Write to the dataset file
            logging_csv(number, mode, pre_processed_landmark_list)

            # Hand sign classification
            #hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

            hand_sign_id = Dtree.predict((np.array(pre_processed_landmark_list)).reshape(1, 42))
            
            hand_sign_id = hand_sign_id.item()

            print(key)
            if mode == 0 and key == 32:      # space
                if (len(caption) == 15):
                    caption = ""
                caption = caption + keypoint_classifier_labels[hand_sign_id]
            
            #print(hand_sign_id)
            #continue
            # if hand_sign_id == 2:  # Point gesture
            #     point_history.append(landmark_list[8])
            # else:
            #     point_history.append([0, 0])

            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(
                debug_image,
                brect,
                keypoint_classifier_labels[hand_sign_id]
            )
        # else:
        # point_history.append([0, 0])

        #debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        #Inserting caption
        debug_image = cv.rectangle(debug_image, (int(frame_width/4), frame_height - 50), (int(3*frame_width/4), frame_height) , (0,0,0), -1)
        debug_image = cv.putText(debug_image, caption, (int(frame_width/4) + 5, frame_height - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Screen reflection #############################################################
        cv.imshow("Inference Video", debug_image)
        #print(caption)

        # 4. Post-process inference results
        #selected_indices = tf_utils.postprocess_for_yolov8(output_boxes, output_scores)

        # 5. Display processed image
        # for idx in selected_indices:
        #     cv.rectangle(
        #         img=resized_frame,
        #         pt1=(int(output_boxes[idx][0]), int(output_boxes[idx][1])),
        #         pt2=(int(output_boxes[idx][2]), int(output_boxes[idx][3])),
        #         color=(0, 255, 0),
        #         thickness=2,
        #     )
        #     cv.putText(
        #         img=resized_frame,
        #         text=class_names[int(output_classes[idx])],
        #         org=(int(output_boxes[idx][0]), int(output_boxes[idx][1])),
        #         fontFace=cv.FONT_HERSHEY_TRIPLEX,
        #         fontScale=0.5,
        #         color=(0, 255, 0),
        #         thickness=1,
        #     )


        #cv.imshow("Inference Video", resized_frame)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    # 6. Cleanup resources
    camera.release()
    cv.destroyAllWindows()


def preprocess_for_mediapipe(frame):
    # Resize image
    resized_frame = cv.resize(frame, (MEDIAPIPE_HEIGHT, MEDIAPIPE_WIDTH))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.float32)
    # Add a batch dimension
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    # Normalize the pixel values to the range [0, 1]
    input_tensor = input_tensor / 255.0

    return input_tensor, resized_frame

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint_akshay.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    return

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def calc_landmark_list_new(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_landmark_akshay(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    # base_x, base_y = 0, 0
    # for index, landmark_point in enumerate(temp_landmark_list):
    #     if index == 0:
    #         base_x, base_y = landmark_point[0], landmark_point[1]

    #     temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
    #     temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    distance = []
    distance.append(landmark_list[1][0] - landmark_list[0][0])
    distance.append(landmark_list[1][1] - landmark_list[1][1])
    distance.append(landmark_list[2][0] - landmark_list[1][0])
    distance.append(landmark_list[2][1] - landmark_list[1][1])
    distance.append(landmark_list[3][0] - landmark_list[2][0])
    distance.append(landmark_list[3][1] - landmark_list[2][1])
    distance.append(landmark_list[4][0] - landmark_list[3][0])
    distance.append(landmark_list[4][1] - landmark_list[3][1])
    distance.append(landmark_list[6][0] - landmark_list[5][0])
    distance.append(landmark_list[6][1] - landmark_list[5][1])
    distance.append(landmark_list[7][0] - landmark_list[6][0])
    distance.append(landmark_list[7][1] - landmark_list[6][1])
    distance.append(landmark_list[8][0] - landmark_list[7][0])
    distance.append(landmark_list[8][1] - landmark_list[7][1])
    distance.append(landmark_list[10][0] - landmark_list[9][0])
    distance.append(landmark_list[10][1] - landmark_list[9][1])
    distance.append(landmark_list[11][0] - landmark_list[10][0])
    distance.append(landmark_list[11][1] - landmark_list[10][1])
    distance.append(landmark_list[12][0] - landmark_list[11][0])
    distance.append(landmark_list[12][1] - landmark_list[11][1])
    distance.append(landmark_list[14][0] - landmark_list[13][0])
    distance.append(landmark_list[14][1] - landmark_list[13][1])
    distance.append(landmark_list[15][0] - landmark_list[14][0])
    distance.append(landmark_list[15][1] - landmark_list[14][1])
    distance.append(landmark_list[16][0] - landmark_list[15][0])
    distance.append(landmark_list[16][1] - landmark_list[15][1])
    distance.append(landmark_list[18][0] - landmark_list[17][0])
    distance.append(landmark_list[18][1] - landmark_list[17][1])
    distance.append(landmark_list[19][0] - landmark_list[18][0])
    distance.append(landmark_list[19][1] - landmark_list[18][1])
    distance.append(landmark_list[20][0] - landmark_list[19][0])
    distance.append(landmark_list[20][1] - landmark_list[19][1])
    distance.append(landmark_list[5][0] - landmark_list[0][0])
    distance.append(landmark_list[5][1] - landmark_list[0][1])
    distance.append(landmark_list[17][0] - landmark_list[0][0])
    distance.append(landmark_list[17][1] - landmark_list[0][1])
    distance.append(landmark_list[5][0] - landmark_list[9][0])
    distance.append(landmark_list[5][1] - landmark_list[9][1])
    distance.append(landmark_list[9][0] - landmark_list[13][0])
    distance.append(landmark_list[9][1] - landmark_list[13][1])
    distance.append(landmark_list[13][0] - landmark_list[17][0])
    distance.append(landmark_list[13][1] - landmark_list[17][1])

    # Normalization
    max_value = max(list(map(abs, distance)))

    def normalize_(n):
        return n / max_value

    distance = list(map(normalize_, distance))

    return distance


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    #info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        #info_text = info_text + ':' + hand_sign_text
        info_text = hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()