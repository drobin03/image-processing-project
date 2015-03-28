import sys
import os
import re
import cv2
import numpy as np
import pdb
import copy
import math

class image:
  def __init__(self):
    self.orig = []
    self.processed = []
    self.grid = []
    self.final = []
    self.output = []
    self.outputGray = []
    self.outputBackup = []

  # Gets the filename from the user
  def getFilename():
    filename = raw_input('Enter a file name: ')
    return filename

  def pre_process(self,img):
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

  def find_grid(self):
    contour_image = copy.copy(self.processed)
    contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest blob
    contours.sort(key=cv2.contourArea)
    biggest = contours.pop()
    return biggest

  # Rectify makes sure that the points of the image are mapped correctly
  # Taken from: http://opencvpython.blogspot.ca/2012/06/sudoku-solver-part-3.html
  def rectify(self,h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

  def fix_perspective(self):
    grid_approx = cv2.approxPolyDP(self.grid,0.01*cv2.arcLength(self.grid,True), True );
    grid_approx = self.rectify(grid_approx)
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
    warp = cv2.warpPerspective(self.orig, M, (maxWidth, maxHeight))
    return warp

  def captureImage(self):
    if (len(sys.argv) > 1):
      filename = sys.argv[1]
    else:
      filename = getFilename()

    # Create a directory to put the resulting images
    result_directory = re.sub("\.[^\.]+$","", filename)
    ext = re.sub("^[^.]+", "", filename)
    if not os.path.exists(result_directory):
      os.makedirs(result_directory)

    self.orig = cv2.imread(filename)

  def processImage(self):
    gray_img = cv2.cvtColor(self.orig, cv2.COLOR_BGR2GRAY)

    # Below are a series of steps to pull the numbers out of a sudoku puzzle
    # I just took the steps from this article: http://www.codeproject.com/Articles/238114/Realtime-Webcam-Sudoku-Solver

    # First: Threshold the image
    self.processed = self.pre_process(gray_img)

    # Second: Detect the grid and crop the original image to contain only the puzzle
    self.grid = self.find_grid()
    # Fix perspective (tilt)
    self.final = self.fix_perspective()
    self.output = np.copy(self.final)
    self.outputBackup = np.copy(self.output)

class OCRmodelClass:
  def __init__(self):
      # samples = np.loadtxt('train/general-samples.data',np.float32)
      # responses = np.loadtxt('train/general-responses.data',np.float32)
      samples = np.loadtxt('generalsamples_mikedeff.data',np.float32)
      responses = np.loadtxt('generalresponses_mikedeff.data',np.float32)
      responses = responses.reshape((responses.size,1))

      #.model uses kNearest to perform OCR
      self.model = cv2.KNearest()
      self.model.train(samples,responses)
      #.iterations contains information on what type of morphology to use
      self.iterations = [-1,0,1,2]
      self.lvl = 0 #index of .iterations

  def OCR(self,image,puzzle):
      #preprocessing for OCR
      #convert image to grayscale
      gray = cv2.cvtColor(image.output, cv2.COLOR_BGR2GRAY)
      #noise removal with gaussian blur
      gray = cv2.GaussianBlur(gray,(5,5),0)
      image.outputGray = gray

      image.output = np.copy(image.outputBackup)
      self.OCR_read(image,puzzle,0)

  def OCR_read(self,image,puzzle,morphology_iteration):
    #perform actual OCR using kNearest model
    thresh = cv2.adaptiveThreshold(image.outputGray,255,1,1,7,2)
    if morphology_iteration >= 0:
        morph = cv2.morphologyEx(thresh,cv2.MORPH_ERODE,None,iterations = morphology_iteration)
    elif morphology_iteration == -1:
        morph = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,None,iterations = 1)

    thresh_copy = morph.copy()
    #thresh2 changes after findContours
    contours,hierarchy = cv2.findContours(morph,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = thresh_copy

    squareWidth = len(image.output)/9
    squareWidth = len(image.output)/9

    # testing section
    for cnt in contours:
      # if cv2.contourArea(cnt) > min_squareArea and cv2.contourArea(cnt) < max_squareArea:
      if cv2.contourArea(cnt) > 20:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20 and h<50 and w>8 and w<50:
          if w<20:
              diff = 20-w
              x -= diff/2
              w += diff
          sudox = x/squareWidth
          sudoy = y/squareWidth
          cv2.rectangle(image.output,(x,y),(x+w,y+h),(0,0,255),2)
          #prepare region of interest for OCR kNearest model
          roi = thresh[y:y+h,x:x+w]
          roismall = cv2.resize(roi,(25,35))
          roismall = roismall.reshape((1,875))
          roismall = np.float32(roismall)
          #find result
          retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k = 1)
          #check for read errors
          if results[0][0]!=0:
            string = str(int((results[0][0])))
            if puzzle[sudoy,sudox]==0:
              puzzle[sudoy,sudox] = int(string)
              cv2.putText(image.output,string,(x,y+h),0,1.4,(255,0,0),3)

def main():
  img = image()
  img.captureImage()
  img.processImage()

  # Fourth: Grab the numbers (This article may be helpful: http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-extracting-digits/)
  reader = OCRmodelClass()
  puzzle = np.zeros((9,9),np.uint8)
  reader.OCR(img,puzzle)
  print puzzle

  cv2.drawContours(img.orig, [img.grid],-1,(0,255,0),3)
  # cv2.imshow("Pre-processed", img.processed)
  # cv2.imshow("Grid", img.orig)
  # cv2.imshow("Perspective Adjusted", img.final)
  cv2.imshow('OCR result',img.output)
  cv2.waitKey(0)
if __name__ == "__main__":
    main()