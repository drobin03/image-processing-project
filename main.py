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
    self.solved = []

  # Gets the filename from the user
  def getFilename(self):
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
    contours.sort(key=cv2.contourArea, reverse=True)
    biggest = None
    for i in contours:
      peri = cv2.arcLength(i,True)
      approx = cv2.approxPolyDP(i,0.02*peri,True)

      # length = approx[]
      if len(approx)==4:
        # Check if it is close to a rectangular shape
        grid_approx = self.rectify(approx)
        (tl, tr, br, bl) = grid_approx
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
        wVariance = abs(widthA-widthB)
        heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))
        hVariance = abs(heightA-heightB)
        if wVariance < .2*widthA and hVariance < .2*heightA:
          biggest = i
          break

    # cv2.drawContours(self.orig, [biggest],-1,(0,255,0),3)
    # cv2.imshow("test", self.orig)
    # # cv2.imshow("test", contour_image)
    # cv2.waitKey(0)
    return biggest


    # self.biggest = None
    #   self.maxArea = 0
    #   for i in self.contours:
    #     area = cv2.contourArea(i)
    #     if area > 50000: #50000 is an estimated value for the kind of blob we want to evaluate
    #       peri = cv2.arcLength(i,True)
    #       approx = cv2.approxPolyDP(i,0.02*peri,True)
    #       if area > self.maxArea and len(approx)==4:
    #         self.biggest = approx
    #         self.maxArea = area
    #         best_cont = i

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
      filename = self.getFilename()

    # Create a directory to put the resulting images
    result_directory = re.sub("\.[^\.]+$","", filename)
    ext = re.sub("^[^.]+", "", filename)
    if not os.path.exists(result_directory):
      os.makedirs(result_directory)

    self.orig = cv2.imread(filename)

    return result_directory

  def processImage(self):
    # Resize to 450px wide
    r = 450.0 / self.orig.shape[1]
    dim = (450, int(self.orig.shape[0] * r))
    # perform the actual resizing of the image and show it
    self.orig = cv2.resize(self.orig, dim, interpolation = cv2.INTER_AREA)

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
    self.solved = np.copy(self.final)

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

class puzzleClass:
  def __init__(self):
    self.model = np.zeros((9,9),np.uint8)
    self.original = []
    self.string_model = ""

  def draw(self,img):
    squareHeight = len(img)/9
    squareWidth = len(img[0])/9
    for i, row in enumerate(self.model):
      for j, num in enumerate(row):
        if (self.original[i][j] == 0):
          posX = (j*squareWidth) + squareWidth/2
          posY = (i*squareHeight) + squareHeight - 10 # padding
          cv2.putText(img,str(num),(posX,posY),0,1.4,(255,0,0),3)

  # Solve method modified from this code: https://freepythontips.wordpress.com/2013/09/01/sudoku-solver-in-python/
  def solve(self):
    self.original = np.copy(self.model)
    self.to_string()
    self.row(list(self.string_model))
    return self.to_array()

  def to_array(self):
    s = self.string_model
    self.model = np.array([[ s[0],  s[1],  s[2],  s[3],  s[4],  s[5],  s[6],  s[7],  s[8] ],
                           [ s[9],  s[10], s[11], s[12], s[13], s[14], s[15], s[16], s[17] ],
                           [ s[18], s[19], s[20], s[21], s[22], s[23], s[24], s[25], s[26] ],
                           [ s[27], s[28], s[29], s[30], s[31], s[32], s[33], s[34], s[35] ],
                           [ s[36], s[37], s[38], s[39], s[40], s[41], s[42], s[43], s[44] ],
                           [ s[45], s[46], s[47], s[48], s[49], s[50], s[51], s[52], s[53] ],
                           [ s[54], s[55], s[56], s[57], s[58], s[59], s[60], s[61], s[62] ],
                           [ s[63], s[64], s[65], s[66], s[67], s[68], s[69], s[70], s[71] ],
                           [ s[72], s[73], s[74], s[75], s[76], s[77], s[78], s[79], s[80] ]],np.uint8)
    return self.model

  def to_string(self):
    self.string_model = ""
    for row in self.model:
      for col in row:
        self.string_model = self.string_model + str(col)

  def row(self, a):
    try:
      i = a.index('0')

      excluded_numbers = set()
      for j in range(81):
        if self.same_row(i,j) or self.same_col(i,j) or self.same_block(i,j):
          excluded_numbers.add(a[j])

      for m in '123456789':
        if m not in excluded_numbers:
          attempt = copy.copy(a)
          attempt[i] = m
          self.row(attempt)

    except ValueError:
      # Solved! Save the puzzle
      self.string_model = ''.join(a)

  def same_row(self,i,j): return (i/9 == j/9)
  def same_col(self,i,j): return (i-j) % 9 == 0
  def same_block(self,i,j): return (i/27 == j/27 and i%9/3 == j%9/3)

def main():
  img = image()
  result_directory = img.captureImage()
  if img.orig == None:
    sys.exit(0)
  img.processImage()

  # Fourth: Grab the numbers (This article may be helpful: http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-extracting-digits/)
  reader = OCRmodelClass()
  puzzle = puzzleClass()
  reader.OCR(img,puzzle.model)

  # Now solve!
  solved = puzzle.solve()
  puzzle.draw(img.solved)
  print puzzle.model

  cv2.drawContours(img.orig, [img.grid],-1,(0,255,0),3)
  cv2.imwrite(result_directory+"/Pre-processed.jpg", img.processed)
  cv2.imwrite(result_directory+"/Grid.jpg", img.orig)
  cv2.imwrite(result_directory+"/PerspectiveAdjusted.jpg", img.final)
  cv2.imwrite(result_directory+"/OCR result.jpg",img.output)
  cv2.imwrite(result_directory+"/Solved.jpg",img.solved)
  #cv2.waitKey(0)
if __name__ == "__main__":
    main()