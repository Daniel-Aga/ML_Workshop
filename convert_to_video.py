import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob

IMAGES_EXT = 'JPG'

parser = argparse.ArgumentParser(description='This program turns a list of .JPG/.jpg files to a .avi video.')
parser.add_argument('--input', type=str, help='Path to a sequence of images.')
parser.add_argument('--output', type=str, help='Name of created video.', default='project1.avi')

args = parser.parse_args()
size = (0,0)

img_array = []
for filename in sorted(glob.glob(f'{args.input}/*.{IMAGES_EXT}')):
  image_path = filename
  
  print(image_path)

  #read image
  img = cv2.imread(image_path)
  height, width, layers = img.shape
  size = (width,height)
  img_array.append(img)
 
#initialize video
out = cv2.VideoWriter(args.output ,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#write video
for i in range(len(img_array)):
  out.write(img_array[i])
out.release()
