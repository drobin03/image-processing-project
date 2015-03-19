import sys
import os
import re
import cv2

# Gets the filename from the user
def getFilename():
  filename = raw_input('Enter a file name: ')
  return filename

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

if __name__ == "__main__":
    main()