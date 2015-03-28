'''
train
Taken from: http://notes.yosemitebandit.com/ocr-with-opencv-and-python/
open a training image of numbers, train.png
preprocess and find contours
for each contour, prompt user for input as to which number is being displayed

annotated from Abid Rahman's post: http://stackoverflow.com/a/9620295/232638
'''
import numpy as np
import cv2

# open training image for processing
image = cv2.imread('train.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# create black and white image
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# find contorus
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST
        , cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0, 100))
responses = []
# keyboard mappings for 0-9; user may type in this range when prompted
keys = [i for i in range(48, 58)]

for contour in contours:
    if cv2.contourArea(contour) > 50:
        # sufficiently large contour to possibly be a number
        [x, y, w, h] = cv2.boundingRect(contour)

        if h > 28:
            # tall enough to possibly be a number
            # draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi = thresh[y:y+h, x:x+w]
            roi_small = cv2.resize(roi, (10, 10))

            # show image and wait for keypress
            cv2.imshow('norm', image)
            key = cv2.waitKey(0)

            if key == 27:
                sys.exit()
            elif key in keys:
                # save pixel data in 1x100 matrix of 'samples'
                sample = roi_small.reshape((1,100))
                samples = np.append(samples,sample,0)
                # save input in 'responses'
                responses.append(int(chr(key)))

print "training complete"
np.savetxt('general-samples.data', samples)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('general-responses.data', responses)