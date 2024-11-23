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
from tkinter import *
from PIL import Image, ImageTk 
import enchant
from fast_autocomplete import AutoComplete, autocomplete_factory


from utils import CvFpsCalc
from model import GestureClassifier

MEDIAPIPE_HEIGHT = 256
MEDIAPIPE_WIDTH = 256
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SCORE_THRESHOLD = 0.90
key = -1

def main():
    use_brect = True
    mode = 0
    word_sug_list = []

    # 0. Initialize
    MODEL_PATH_LANDMARK = "C:\AI-buzz\project_main\model\mediapipe_hand-mediapipehandlandmarkdetector.tflite"
    MODEL_GESTURE = "C:\AI-buzz\project_main\DT_gesture_model.pkl"
    #LABEL_PATH = utils.get_label_path()

    gesture_recognition_model = GestureClassifier()
    Dtree = joblib.load(MODEL_GESTURE)

    #contains the labels
    with open('model/gesture_classifier/gesture_classifier_label_akshay.csv',
              encoding='utf-8-sig') as f:
        gesture_classifier_labels = csv.reader(f)
        gesture_classifier_labels = [
            row[0] for row in gesture_classifier_labels
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
    interpreter_landmark = tf.lite.Interpreter(model_path=MODEL_PATH_LANDMARK, num_threads = 4)
    interpreter_landmark.allocate_tensors()

    # Get input and output tensors for landmark detection detection.
    input_details = interpreter_landmark.get_input_details()
    output_details = interpreter_landmark.get_output_details()

    caption = ""
    word = ""

    with open('C:/Users/agola/Downloads/google-10000-english.txt') as word_file:
        valid_words = list(word_file.read().split())

    valid_words_dict = {}
    for valid_word in valid_words:
        valid_words_dict[valid_word] = {}

    # Create a GUI app 
    app = Tk() 

    # Bind the app with Escape keyboard to quit app whenever pressed 
    app.bind('<Escape>', lambda e: app.quit())
    app.bind('<KeyPress>', lambda e: update_key(e))
  
    # Create a label and display it on app 
    label_widget = Label(app) 
    label_widget.pack()

    button_frame = Frame(app, width=200, height=100)

    cor_btn = Button(button_frame, text = "spellcheck_but", height=2 )
    sug_btn1 = Button(button_frame, text = "sug_but1", height=2)
    sug_btn2 = Button(button_frame, text = "sug_but2", height=2)

    def auto_comp(word_):
        autocomplete = AutoComplete(words = valid_words_dict)
        word_sug_list = autocomplete.search(word=word_, max_cost=3, size=3)
        return word_sug_list

    # 1. Capture image from camera
    def open_camera_():

        nonlocal mode
        global key
        nonlocal caption
        nonlocal word
        nonlocal word_sug_list
        nonlocal cor_btn
        nonlocal sug_btn1
        nonlocal sug_btn2

        fps = cvFpsCalc.get()
        number, mode = change_mode(key, mode)

        success, frame = camera.read()

        if not success:
            return
        frame_flip = cv.flip(frame, 1)  # Mirror display
        debug_frame = copy.deepcopy(frame_flip)

        # 2. Pre-process image
        #input_tensor, resized_frame = tf_utils.preprocess_for_yolov8(frame)
        input_tensor, resized_frame = preprocess_for_mediapipe(debug_frame)

        # 3. Execute model and get inference results

        # Run the model on input tensor.
        interpreter_landmark.set_tensor(input_details[0]["index"], input_tensor)
        interpreter_landmark.invoke()

        score = interpreter_landmark.get_tensor(output_details[0]["index"])  # .reshape(2944, 18) #  if there is a hand or not.
        lr = interpreter_landmark.get_tensor(output_details[1]["index"]) # .reshape(2944) #if its left or right hand
        hand_landmarks = interpreter_landmark.get_tensor(output_details[2]["index"]).reshape(21, 3) #hand coordinates

        if (key == 99):            #press c for backspace
            caption = caption[:-1]
            word = word[:-1]
            key = -1

        if score > SCORE_THRESHOLD:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_frame, hand_landmarks)
            # Landmark calculation
            # transforms coordinates to a list and discards Z coordinates. 
            landmark_list = calc_landmark_list_new(debug_frame, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark_akshay(
                landmark_list)

            # Write to the dataset file
            #logging_csv(number, mode, pre_processed_landmark_list)

            hand_sign_id = Dtree.predict((np.array(pre_processed_landmark_list)).reshape(1, 42))
            
            hand_sign_id = hand_sign_id.item()

            def update_word(word_):
                nonlocal caption
                nonlocal word
                caption = caption[:-len(word)]
                caption = caption + word_
                word = word_

            print(key, word)      # remove during submission
            if mode == 0 and key != -1:      # space
                if (len(caption) == 14):
                    caption = word
                if (key == 98):
                    caption = caption + " "
                    word = ""
                elif  (key == 32):
                    caption = caption + gesture_classifier_labels[hand_sign_id]
                    word = word + gesture_classifier_labels[hand_sign_id]
                key = -1
                used_word_list = []        # list to avoid repititions
                try:
                    word_sug_list = check_word_spell(word)
                    #print(word_sug_list[0])
                    cor_btn['text'] = word_sug_list[0].upper()
                    cor_btn.configure(command=lambda: update_word(cor_btn['text']))
                    used_word_list.append(cor_btn['text'])
                except:
                    pass
                try:
                    word_cor_list = auto_comp(word)
                    for word_cor in word_cor_list:
                        if (word_cor[0].upper() not in used_word_list):
                            sug_btn1['text'] = word_cor[0].upper()
                            break
                    sug_btn1.configure(command=lambda: update_word(sug_btn1['text']))
                    used_word_list.append(sug_btn1['text'])
                except:
                    pass
                try:
                    #word_cor_list = auto_comp(word)
                    for word_cor in word_cor_list:
                        if (word_cor[0].upper() not in used_word_list):
                            print("label1", word_cor[0].upper())
                            sug_btn2['text'] = word_cor[0].upper()
                            break
                    sug_btn2.configure(command=lambda: update_word(sug_btn2['text']))
                except:
                    pass

            cor_btn.pack(side = LEFT, expand = True, fill = BOTH)
            sug_btn1.pack(side = LEFT, expand = True, fill = BOTH)  
            sug_btn2.pack(side = LEFT, expand = True, fill = BOTH)              

            # Drawing part
            debug_frame = draw_bounding_rect(use_brect, debug_frame, brect)
            debug_frame = draw_landmarks(debug_frame, landmark_list)   #uncomment to include landmark in frame
            debug_frame = draw_info_text(
                debug_frame,
                brect,
                gesture_classifier_labels[hand_sign_id]
            )

        debug_frame = draw_info(debug_frame, fps, mode, number)

        #Inserting caption
        debug_frame = cv.rectangle(debug_frame, (int(frame_width/4), frame_height - 50), (int(3*frame_width/4), frame_height) , (0,0,0), -1)
        debug_frame = cv.putText(debug_frame, caption, (int(frame_width/4) + 5, frame_height - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)


        # Screen reflection #############################################################
        #cv.imshow("Inference Video", debug_frame)

        #cv.displayOverlay("Inference Video", caption, 0)

        # Convert image from one color space to other 
        opencv_image = cv.cvtColor(debug_frame, cv.COLOR_BGR2RGBA) 
  
        # Capture the latest frame and transform to image 
        captured_image = Image.fromarray(opencv_image) 
  
        # Convert captured image to photoimage 
        photo_image = ImageTk.PhotoImage(image=captured_image) 
  
        # Displaying photoimage in the label 
        label_widget.photo_image = photo_image

        button_frame.pack(side = BOTTOM)
  
        # Configure image in the label 
        label_widget.configure(image=photo_image) 
        
        # Repeat the same process after every 10 seconds 
        label_widget.after(20, open_camera_) 

        # key = cv.waitKey(20)
        # if key & 0xFF == ord("d"):
        #     break

    # Create a button to open the camera in GUI app 
    camera_button = Button(app, text="Open Camera", command=open_camera_) 
    camera_button.pack(side = BOTTOM, expand = True, fill = BOTH)
  
    # Create an infinite loop for displaying app on screen 
    app.mainloop()

def check_word_spell(word):
    # Using 'en_US' dictionary 
    d = enchant.Dict("en_US")
  
    d.check(word)

    # Will suggest similar words from given dictionary 
    return d.suggest(word)

def update_key(event):
    global key
    key = ord(event.char)
    

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
        csv_path = 'model/gesture_classifier/gesture_akshay.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
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

    rel_dist = []
    rel_dist.append(landmark_list[1][0] - landmark_list[0][0])
    rel_dist.append(landmark_list[1][1] - landmark_list[1][1])
    rel_dist.append(landmark_list[2][0] - landmark_list[1][0])
    rel_dist.append(landmark_list[2][1] - landmark_list[1][1])
    rel_dist.append(landmark_list[3][0] - landmark_list[2][0])
    rel_dist.append(landmark_list[3][1] - landmark_list[2][1])
    rel_dist.append(landmark_list[4][0] - landmark_list[3][0])
    rel_dist.append(landmark_list[4][1] - landmark_list[3][1])
    rel_dist.append(landmark_list[6][0] - landmark_list[5][0])
    rel_dist.append(landmark_list[6][1] - landmark_list[5][1])
    rel_dist.append(landmark_list[7][0] - landmark_list[6][0])
    rel_dist.append(landmark_list[7][1] - landmark_list[6][1])
    rel_dist.append(landmark_list[8][0] - landmark_list[7][0])
    rel_dist.append(landmark_list[8][1] - landmark_list[7][1])
    rel_dist.append(landmark_list[10][0] - landmark_list[9][0])
    rel_dist.append(landmark_list[10][1] - landmark_list[9][1])
    rel_dist.append(landmark_list[11][0] - landmark_list[10][0])
    rel_dist.append(landmark_list[11][1] - landmark_list[10][1])
    rel_dist.append(landmark_list[12][0] - landmark_list[11][0])
    rel_dist.append(landmark_list[12][1] - landmark_list[11][1])
    rel_dist.append(landmark_list[14][0] - landmark_list[13][0])
    rel_dist.append(landmark_list[14][1] - landmark_list[13][1])
    rel_dist.append(landmark_list[15][0] - landmark_list[14][0])
    rel_dist.append(landmark_list[15][1] - landmark_list[14][1])
    rel_dist.append(landmark_list[16][0] - landmark_list[15][0])
    rel_dist.append(landmark_list[16][1] - landmark_list[15][1])
    rel_dist.append(landmark_list[18][0] - landmark_list[17][0])
    rel_dist.append(landmark_list[18][1] - landmark_list[17][1])
    rel_dist.append(landmark_list[19][0] - landmark_list[18][0])
    rel_dist.append(landmark_list[19][1] - landmark_list[18][1])
    rel_dist.append(landmark_list[20][0] - landmark_list[19][0])
    rel_dist.append(landmark_list[20][1] - landmark_list[19][1])
    rel_dist.append(landmark_list[5][0] - landmark_list[0][0])
    rel_dist.append(landmark_list[5][1] - landmark_list[0][1])
    rel_dist.append(landmark_list[17][0] - landmark_list[0][0])
    rel_dist.append(landmark_list[17][1] - landmark_list[0][1])
    rel_dist.append(landmark_list[5][0] - landmark_list[9][0])
    rel_dist.append(landmark_list[5][1] - landmark_list[9][1])
    rel_dist.append(landmark_list[9][0] - landmark_list[13][0])
    rel_dist.append(landmark_list[9][1] - landmark_list[13][1])
    rel_dist.append(landmark_list[13][0] - landmark_list[17][0])
    rel_dist.append(landmark_list[13][1] - landmark_list[17][1])

    # Normalization
    max_value = max(list(map(abs, rel_dist)))

    def normalize_(n):
        return n / max_value

    rel_dist = list(map(normalize_, rel_dist))

    return rel_dist


def change_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 116:  # t for training
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
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
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