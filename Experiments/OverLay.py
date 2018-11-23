import os
from PIL import Image
import numpy as np
from matplotlib import image as mplimg

Segm_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTube_Segmentation/final_DB/"
Orig_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTubeFaces/frame_images_DB/"
Overlay_path = "/mnt/data/users/raj/Raj/IndividualProject/YouTube_Segmentation/Overlay/"

for _, persons, _ in os.walk(Segm_path):
  persons.sort()
  for person in persons:
    print("Performing blending for:", person)
    for _, videos, _ in os.walk(Segm_path + person):
      videos.sort()
      for video in videos:
        vid_path = Segm_path + person + "/" + video
        out_vid_path = Overlay_path + person + "/" + video
        if os.path.exists(out_vid_path):
          continue
        else:
          os.makedirs(out_vid_path)
        for path, _, files in os.walk(vid_path):
          files.sort()
          for filename in files:
            segm = np.array(Image.open(vid_path + "/" + filename))
            image = Image.open(Orig_path + person + "/" + video + "/" + filename)
            
            
            segm = Image.fromarray(np.uint16(segm[:, :, 0]))
            segm = segm.convert("RGB")
            out = Image.blend(image, segm, 0.4)
            mplimg.imsave(out_vid_path + "/" + filename, out)            

    
  
