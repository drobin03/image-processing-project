import sys
import os
import re
import cv2
import numpy as np
import pdb
import copy

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
  contour_image = copy.copy(img)
  contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest blob
  contours.sort(key=cv2.contourArea)
  biggest = contours.pop()
  return biggest

# Rectify makes sure that the points of the image are mapped correctly
# Taken from: http://opencvpython.blogspot.ca/2012/06/sudoku-solver-part-3.html
def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]

  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

def fix_perspective(grid, img):
  grid_approx = cv2.approxPolyDP(grid,0.01*cv2.arcLength(grid,True), True );
  grid_approx = rectify(grid_approx)
  (tl, tr, br, bl) = grid_approx
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))

  # ...and now for the height of our new image
  heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))

  # take the maximum of the width and height values to reach
  # our final dimensions
  maxWidth = max(int(widthA), int(widthB))
  maxHeight = max(int(heightA), int(heightB))

  # construct our destination points which will be used to
  # map the screen to a top-down, "birds eye" view
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

  # calculate the perspective transform matrix and warp
  # the perspective to grab the screen
  M = cv2.getPerspectiveTransform(grid_approx, dst)
  warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
  return warp

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

  img = cv2.imread(filename)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Below are a series of steps to pull the numbers out of a sudoku puzzle
  # I just took the steps from this article: http://www.codeproject.com/Articles/238114/Realtime-Webcam-Sudoku-Solver

  # First: Threshold the image
  processed = pre_process(gray_img)

  # Second: Detect the grid and crop the original image to contain only the puzzle
  grid = find_grid(processed)
  cv2.drawContours(img, [grid],-1,(0,255,0),3)
  # Fix perspective (tilt)
  grid_adjusted = fix_perspective(grid, gray_img)

  # Fourth: Grab the numbers (This article may be helpful: http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-extracting-digits/)

  cv2.imshow("Pre-processed", processed)
  cv2.imshow("Grid", img)
  cv2.imshow("Perspective Adjusted", grid_adjusted)
  cv2.waitKey(0)
if __name__ == "__main__":
    main()