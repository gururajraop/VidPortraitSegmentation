import os
from PIL import Image
import numpy as np
from matplotlib import image as mplimg

Segm_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTube_Segmentation/final_DB/"
#Orig_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTubeFaces/frame_images_DB/"
#Overlay_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTube_Segmentation/Overlay/"
Out_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTube_Segmentation/ConvertedSegm/"

for _, persons, _ in os.walk(Segm_path):
  persons.sort()
  for person in persons:
    print("Performing conversion for:", person)
    for _, videos, _ in os.walk(Segm_path + person):
      videos.sort()
      for video in videos:
        vid_path = Segm_path + person + "/" + video
        out_vid_path = Out_path + person + "/" + video
        if os.path.exists(out_vid_path):
          continue
        else:
          os.makedirs(out_vid_path)
        for path, _, files in os.walk(vid_path):
          files.sort()
          for filename in files:
            segm = np.array(Image.open(vid_path + "/" + filename))
            filename = filename.replace(".jpg", ".png")

            out = segm[:, :, 0]
            out = (out == 253).astype(int)
            label = Image.fromarray(np.uint32(out))

            label.save(out_vid_path + "/" + filename)
