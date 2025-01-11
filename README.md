## Implementation Summary
This project was developed to train and recognise the symbols in ASL (American Sign Language). As a starting point we have focused on training and recognising the **ASL alphabets** using the Qualcomm's MediaPipe-Hand-Detection model. This model is optimized for Real‑time hand detection on mobile and edge. When running the application, the user can make the gesture corresponding to an alphabet and the application recognizes it. The symbols can be recorded to construct sentences with the help of auto-suggestion and spell-check.

### app.py
This file is the starting point of the application. It calls and imports the necessary libraries and the pretrained model. 

#### open_camera()
This function is responsible for the overall flow of the application. It captures the images from the camera and inputs it to the Mediapipe model. It takes the output of the Mediapipe model and inputs it to the preprocessing function. The output from the preprocessing model is then fed to the pretrained Decision Tree model to predict the gesture.

This function is also responsible for annotating the recognised symbol. It is used to take input from the keyboard to record/clear the recognised alphabets.

#### pre_process_landmark()
This function takes the output of the MediaPipe model (21 points with x and y coordinates) and calculates the relative distances between different landmark points. It normalizes the distances and outputs the same. 

#### auto_complete() and checkwordspell()
These functions use the AutoComplete() and the Enchant() function for word suggestion and spellcheck based on the current word.

#### update_word()
Updates the current word in the caption with the word chosen by the user in the UI. 

### Limitations
The model struggles in recognising the symbol when the hand is turned back (palm not visible). This made it tough to train the symbols for alphabets like 'M' and 'N'. 

The transition between two symbols is not captured well by the model. The model takes time to recognise the correct landmark points when the second symbol is made by the user. 

Better pre-processing is required to get a clear distinction between symbols. **Apart from relative distance, metrics like 'Finger closed', 'Finger curved', 'Angle of tilt' can be used during preprocessing to make the model more accurate.**

### Future scope
Pre-processing will be modified and enhanced to precisely predict the gesture made. Transition between two symbols will be handled gracefully. 

Our goal is to train the model for **recognising static gestures in ASL**. We plan to develop an Android app and **gamify the approach towards learning ASL**. This will make it easy and help parents train their kids.

We also plan to provide the option for **training custom symbols**. This can help children in age of 3-5yrs old to understand their environment and interact by referring to it. 

## Installation & Setup steps
<!-- 
Mention in detail how a reviewer can install and run your project. Prefereable include a script to automate the setup.
Make sure to include the pre-requisite packages/assumptions (e.g. Java, Android Studio) in detail.
-->

1) Install Python >= 3.8
2) Create a virtual environment.
3) Install the dependencies mentioned in the requirements.txt file using pip install -r requirements.txt
4) Run the app.py file. 

## Expected output/behaviour (Instructions to use)
<!-- 
Provide details of expected behaviour and output.
Mention how the reviewer can validate the prototype is doing what it is intended to.
If your prototype requires some files / data for evaluation, make sure to provide the files along with instructions on using them.
-->

Run the app.py script using "python app.py"
This opens the camera and the frames are read. 

Note: Ensure your camera is opened

Choose a word to record. Eg: "FLOW"
1) Take reference from the attached image and make the gesture from your right hand of 1st letter of the word. When the alphabet is recognised, press the spacebar to record the alphabet. 
2) Continue to make the gestures of the subsequent letters and record them. 
3) Auto complete and Spell check suggestions are shown in the buttons below the caption. Click on the button with the intended word to record it.

Note: 
1) Click the spacebar to record the recognised letter.
2) To record a space, input letter 'B' from the keyboard. 
3) To clear the last alphabet, input letter 'C' from the keyboard

!(ASL.jpg)

## Any additional steps required for running
- [x] NA
<!-- 
Mention any additional requirements here. If not, leave the NA.
-->

## Submission Checklist
- [x] Recorded video
- [x] Readme updated with required fields
- [x] Dependency installation scripts added
- [ ] Startup script added
- [x] Idea url updated in Readme
