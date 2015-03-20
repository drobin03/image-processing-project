import sys
import os
import re
import cv2
import numpy as np
import pdb

# Gets the filename from the user
def getFilename():
  filename = raw_input('Enter a file name: ')
  return filename

def pre_process(img):
  medBlur = cv2.medianBlur(img,5)
  thresholded = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) # I took these values straight from the opencv docs, they could be played with (http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
  # invert the image
  cv2.bitwise_not(thresholded, thresholded)
  # connect disconnected lines
  kernel = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], np.uint8)
  thresholded_dilated = cv2.dilate(thresholded, kernel)
  return thresholded_dilated

def find_grid(img):
  contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest blob
  biggest = None
  max_area = 0
  for i in contours:
    area = cv2.contourArea(i)
    if area > 100:
      peri = cv2.arcLength(i,True)
      approx = cv2.approxPolyDP(i,0.02*peri,True)
      if area > max_area and len(approx)==4:
        biggest = approx
        max_area = area
  # contours.sort(key=cv2.contourArea, reverse=True)
  # biggest = contours.pop()

  # Now crop the image around the grid
  # TODO
  return biggest

def main():
  if (len(sys.argv) > 1):
    filename = sys.argv[1]
  else:
    filename = getFilename()

  # Create a directory to put the resulting images
  result_directory = re.sub("\.[^\.]+$","", filename)
  ext = re.sub("^[^.]+", "", filename)
  if not os.path.exists(result_directory):
    os.makedirs(result_directory)

  img = cv2.imread(filename,0)

  # Below are a series of steps to pull the numbers out of a sudoku puzzle
  # I just took the steps from this article: http://www.codeproject.com/Articles/238114/Realtime-Webcam-Sudoku-Solver

  # First: Threshold the image (Convert to black and white)
  processed = pre_process(img)

  grid = find_grid(processed)

  cv2.imshow("Pre-processed", processed)
  cv2.waitKey(0)

  # Second: Rotate the image so that it is straight up and down

  # Third: Detect the grid lines

  # Fourth: Grab the numbers (This article may be helpful: http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-extracting-digits/)

if __name__ == "__main__":
    main()