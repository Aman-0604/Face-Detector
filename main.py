# importing libraries
import os
import cv2
import matplotlib.pyplot as plt

# defining two directories
data_dir = './data'
models_dir = './models'

# creating my face detector
face_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))

# iterate through all the images
flag = 0
for img_path in os.listdir(data_dir):
    # load the image I am gonna use
    if flag == 0:
        flag = 1
        continue
    img = cv2.imread(os.path.join(data_dir, img_path))

    # covert this image to grayscale, so I will create another image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # a list containing all the faces we detected so far with our face detector
    # we set one of its parameters to 20 (parameter tuning) so that errors shall go
    faces = face_detector.detectMultiScale(img_gray, minNeighbors=20)

    # build all the plots in a new window
    plt.figure()

    # iterating through all the face in the faces list
    for face in faces:
        # unwrap all the parameters for this face which is actually a rectangle
        # x and y are the upper left coordinates of the 'face' not 'image'
        # width and height are of the 'face' not of the 'image'
        x, y, width, height = face

        # call a rectangle because I am going to draw a bounding box on our image so we can know what are the
        # faces we are detecting
        # @parameters: (img, upper left coordinate, bottom right coordinate, color, thickness value)
        img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 10)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plot this image
    plt.imshow(img_rgb)

plt.show()