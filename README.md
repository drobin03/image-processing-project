# Sudoku Solver

This program takse an image of a Sudoku puzzle, and attempts to find a solution for it.

The /site folder contains code for a website, visible at sudoku-solver.iamdan.me, but the program can be run seperately on the command line for testing.

## How to run the python code

python main.py <image>

- where <image> is the path to the image that you want to solve (There are some test images in site/public/images/)
- The resulting solved image will be output into a subfolder named after the image (ex. site/public/images/sudoku/Solved.jpg)
- The program will also output 4 other images into the resulting directory, showing the steps of the solution