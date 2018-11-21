import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mplimg
from skimage import io


YT_data_path = "/home/raj/data/Raj/IndividualProject/YouTube_Segmentation/final_DB/"
Output_path = "/home/raj/data/Raj/IndividualProject/YouTube_Segmentation/final_DB/"

if True:
  for _, persons, _ in os.walk(YT_data_path):
    persons.sort()
    for person in persons:
      print("Performing segmentation for:", person)
      for _, videos, _ in os.walk(YT_data_path + "/" + person):
        videos.sort()
        for video in videos:
          vid_path = YT_data_path + "/" + person + "/" + video
          out_vid_path = Output_path + "/" + person + "/" + video
          for path, _, files in os.walk(vid_path):
            files.sort()
            for filename in files:
              orig_file = path + "/" + filename
              filename = filename.replace("output_", "")
              new_file = path + "/" + filename

              os.system("mv " + orig_file + " "  + new_file)
