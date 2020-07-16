# Yolov3-320
# Overview
In this project I have downloaded the configuration and weights of yolo version3 - 320  which can detect 80 classes.
Here 320 reffers the input image size. the configured network had 3 output layers
each output is size of (300,85) (1200,85) and (4800,85)
Here in 85 columns first five columns reffers x,y,height,width,confidence score respectively
and next 80 columns reffers to class predicted values
at first i have stored the highest class vlaue with a threshold of 0.5 and by using the first 4 columns i have drawn the bounding boxes around the object.
# Result of an Image
![](res1.jpg)
