import os
from PIL import Image
import numpy as np
from matplotlib import image as mplimg


for _, persons, _ in os.walk("Segm/"):
  persons.sort()
  for person in persons:
    print("Performing blending for:", person)
    for _, videos, _ in os.walk("Segm/" + person):
      videos.sort()
      for video in videos:
        vid_path = "Segm/" + person + "/" + video
        out_vid_path = "Overlap/" + person + "/" + video
        if os.path.exists(out_vid_path):
          continue
        else:
          os.makedirs(out_vid_path)
        for path, _, files in os.walk(vid_path):
          files.sort()
          for filename in files:
            segm = np.array(Image.open(vid_path + "/" + filename))
            filename = filename.replace("output_", "")
            image = Image.open("Orig/" + person + "/" + video + "/" + filename)
            
            
            segm = Image.fromarray(np.uint(segm[:, :, 0]))
            segm = segm.convert("RGB")
            out = Image.blend(image, segm, 0.4)
            mplimg.imsave(out_vid_path + "/" + filename, out)            

    
  
