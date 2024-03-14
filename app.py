import numpy as np
import cv2
import mediapipe as mp
import joblib
import time
import winsound
from random import random
from english_words import get_english_words_set
web2lowerset = get_english_words_set(['web2'], lower=True)
from flask import Flask, render_template, Response, request, flash, redirect, url_for, jsonify

global easy, medium, eraser, done
easy=0
medium=0
eraser=0
switch=1
done = False

clf = joblib.load("random_forest.joblib")


app = Flask(__name__, template_folder='./template')


def camera_max():
    camera = 0
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0,int(camera-1)))
        
cam_max = camera_max()


cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
words = [i for i in sorted(list(web2lowerset)) if 'z' and 'g' and 'h' not in i and len(i) > 3 and len(i) <= 10]
start_time = time.time()
curr_time = 0
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0
location = 0
letter_help = 0
done = False

def practice_mode(frame):
    global cap, done, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = model.process(image)                 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])


    #Main function
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    done = False

    # Set mediapipe model
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():
            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            letter_help = cv2.resize(cv2.imread('practice_mode_letters/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            # print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.7 or (max(test_probs) >= 0.8 and letters[test_pred] in ['p','r','u','v']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                    sound_notification()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                    sound_notification()

                            if easy_word_user == easy_word:
                                done = True
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''
                                sound_end_notification()
                                time.sleep(1)

                        except Exception as e:
                            print(e)

            # Show letter helper
            frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help

            return frame
            
    return frame

def assess_mode(frame):
    global cap, done , easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    done = False
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = model.process(image)                 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])
    
    #Main function
    #cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():
            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            # print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.7 or (max(test_probs) >= 0.8 and letters[test_pred] in ['p','r','u','v']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                    sound_notification()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                    sound_notification()

                            if easy_word_user == easy_word:
                                done = True
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''
                                sound_end_notification()
                                time.sleep(1)


                        except Exception as e:
                            print(e)

            try: 
                letter_help == 0
            except:
                frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5]
            
            return frame

    return frame

def sign_frame():  # generate frame by frame from camera
    global easy, cap, eraser, medium
    while True:
        success, frame = cap.read() 
        if success:
            if(easy):                
                frame = practice_mode(frame)
            elif(medium):
                frame = assess_mode(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

def sound_end_notification():
    duration = 2000
    freq = 400
    winsound.Beep(freq, duration)
    print('end_notification')

def sound_notification():
    duration = 1000
    freq = 400
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

@app.route('/')
def index():
    return render_template("landing-page.html")


@app.route('/video_feed')
def video_feed():
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/learn')
def learn():
    return render_template("learn.html")

@app.route('/requests',methods=['POST','GET'])
def mode():
    print("mode route")
    global switch, easy, medium
    if request.method == 'POST':
        if request.form.get('practice') == 'Practice':
            easy= not easy
            medium =  0
        elif  request.form.get('assess') == 'Assess':
            medium = not medium
            easy = 0                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')