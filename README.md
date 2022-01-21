![sleepyman](https://user-images.githubusercontent.com/82250641/117279065-b6e7a200-ae37-11eb-9f12-4a43f08b3221.jpeg)


# Drowsiness detection
According to the Centers for Disease Control and Prevention (CDC), 1 in 25 adults report they have fallen asleep while driving in the past 30 days, in addition to several studies suggested that about 20% of all road accidents are related to fatigue. 
As a solution, in this project I bring a system that detects drowsiness and sounds an alarm before it is too late.

## Detector
This system detects landmarks in a video frame to extract the necessary characteristics, in this case the Eye Aspect Ratio (EAR)

![1_BlsrFzC8H6i3xv695Ja6vg](https://user-images.githubusercontent.com/82250641/119591720-839c9100-bdad-11eb-8968-78d982365ead.png)

EAR, as the name suggests, is the ratio of the length of the eyes to the width of the eyes. The length of the eyes is calculated by averaging over two distinct vertical lines across the eyes.

![blink_detection_plot](https://user-images.githubusercontent.com/82250641/119591715-826b6400-bdad-11eb-88b5-bf53a2b75c04.jpg)


## Resources used
**Python Version:** 3.8.5

**Packages:** OpenCV, dlib, time, imutils, Scipy, Numpy, Playsound, Matplotlib, Threading 

**Landmarks article (Adrian Rosebrock):** https://bit.ly/3h7hzxs

**Drowsiness detection article (Adrian Rosebrock):** https://bit.ly/3unGxfC

**Academic paper:** https://bit.ly/3nQLmvs
