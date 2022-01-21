# Code based on Adrian Rosebrock's article
# https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/

# import required libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

# define constants
ALARM = "alarm.wav"
WEBCAM = 1
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40
COUNTER = 0
ALARM_ON = False


def sound_alarm(path=ALARM):
    # play an alarm sound
    playsound.playsound(ALARM)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# dlib's face detector (HOG-based)
print("[INFO] loading the landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# take the indexes of the predictor, for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initializing video
print("[INFO] initializing streaming video ...")
vs = VideoStream(src=WEBCAM).start()
time.sleep(1.0)

# draw a figure-like object
y = [None] * 100
x = np.arange(0,100)
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(x, y)

# loop over video frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces (grayscale)
    rects = detector(gray, 0)

    # loop on face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract eye coordinates and calculate the aperture ratio
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # convex hull for eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # save history to plot
        y.pop(0)
        y.append(ear)

        # update canvas
        plt.xlim([0, 100])
        plt.ylim([0, 0.4])
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        plt.show(block=False)
        li.set_ydata(y)
        fig.canvas.draw()
        time.sleep(0.01)

        # check ratio x threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # within the criteria, sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # turn on alarm
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "[ALERT] FATIGUE!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # if above the limit, reset the counter and turn off the alarm
        else:
            COUNTER = 0
            ALARM_ON = False

            # draw the proportion of eye opening
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # key to exit the script "q"
    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.stop()