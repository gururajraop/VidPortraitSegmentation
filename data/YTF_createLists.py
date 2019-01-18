import os
import random

input_path="./YouTubeFaces_Images/"
label_path="./YouTubeFaces_BinaryLabels/"

lists = "./Lists/"
if not os.path.exists(lists):
  os.makedirs(lists)

random.seed(123)

f_train = open(lists + "train.txt","w+")
f_val = open(lists + "val.txt","w+")

train_count = 0
val_count = 0

for _, persons, _ in os.walk(label_path):
  persons.sort()
  for person in persons:
    for _, videos, _ in os.walk(label_path + person):
      videos.sort()
      for video in videos:
        is_val = random.randint(0,11) < 3
        input_vid_path = input_path + person + "/" + video
        label_vid_path = label_path + person + "/" + video
        for path, _, files in os.walk(label_vid_path):
          files.sort()
          for filename in files:
            frame_in_path = input_vid_path + "/" + filename
            frame_in_path = frame_in_path.replace("./", "")
            frame_label_path = label_vid_path + "/" + filename
            frame_label_path = frame_label_path.replace("./", "")
            frame_path = frame_in_path + "," + frame_label_path + "\n"
            if not is_val:
              f_train.write(frame_path)
              train_count += 1
            else:
              f_val.write(frame_path)
              val_count += 1

print("Total frames in train set: ", train_count)
print("Total frames in val set: ", val_count)

f_train.close()
f_val.close()
