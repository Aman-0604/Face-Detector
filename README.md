## Step 1:
Create a file named  `requirements.txt`

Paste these lines in it:
```commandline
matplotlib==3.5.0
opencv-python==4.5.4.60
```
```commandline
pip install -r requirements.txt
```

## Step 2:
import libraries

```commandline
import os
import cv2
import matplotlib.pyplot as plt
```

## Step 3:
Make two directories namely `data` and `models`
In `data` directory we added 3 images with superb frontal face and superb lighting.

In `models` directory paste the code of the model available from opencv

```commandline
https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## Step 4:
Create my face detector by creating a variable name `face_detector`

```commandline
face_detector = cv2.Cascade_Classifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))
```

So now we need to iterate through all the images in the `data` directory and use our face detector on them to see what faces are being detected.

## Step 5:
Iterate through all the images.

1. load the images
2. then convert them to gray scale bcoz the face detector receives the gray scale images as inputs, so i need a gray scale image.
3. create a variable `faces` which will be a list containing all the faces we detected through our face detector(in our data each image contains only 1 face)
4. iterate through all of these faces in the `faces` list
5. plot the image
