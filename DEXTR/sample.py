import os
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mplimg

from torch.nn.functional import upsample

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

from maskRCNN.maskrcnn_benchmark.config import cfg
from maskRCNN.demo.predictor_person import COCODemo
from skimage import io

PAD_SIZE = 10


def maskRCNN_model():
    config_file = "/home/raj/data/Raj/IndividualProject/maskRCNN/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.9,
    )

    return coco_demo

def get_maskRCNN_predictions(model, image_path):
    image = io.imread(image_path)
    predictions, bbox, masks, heatmap = model.run_on_opencv_image(image)

    return predictions, bbox, masks, heatmap



modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cpu")
#device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)

#  Read image and click the points
#plt.ion()
#plt.axis('off')
#plt.imshow(image)
#plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

#results = []

def get_extreme_points(BBox):
    x_min = np.int(BBox[0][0])
    y_min = np.int(BBox[0][1])
    x_max = np.int(BBox[0][2])
    y_max = np.int(BBox[0][3])

    # Mid point
    #top = np.array([(x_max-x_min)/2, y_min])
    #bottom = np.array([(x_max-x_min)/2, y_max])
    #left = np.array([x_min, (y_max-y_min)/2])
    #right = np.array([x_max, (y_max-y_min)/2])

    # Original
    #top = np.array([x_min, y_min])
    #bottom = np.array([x_max, y_max])
    #left = np.array([x_min, y_max])
    #right = np.array([x_max, y_min])

    # Customized
    top = np.array([x_min+(x_max-x_min)*0.5, y_min-PAD_SIZE])
    bottom = np.array([x_min+(x_max-x_min)*0.5, y_max+PAD_SIZE])
    left = np.array([x_min-PAD_SIZE, y_min+(y_max-y_min)*0.95])
    right = np.array([x_max+PAD_SIZE, y_min+(y_max-y_min)*0.95])

    extreme_points = np.array([top, left, right, bottom]).astype(np.int)

    return extreme_points


def get_EP_by_mask(mask):
    mask = mask.squeeze()
    idx = np.nonzero(mask)

    left = [np.min(idx[1]), idx[0][np.argmin(idx[1])]]
    right = [np.max(idx[1]), idx[0][np.argmax(idx[1])]]
    top = [idx[1][np.argmin(idx[0])], np.min(idx[0])]
    bottom = [idx[1][np.argmax(idx[0])], np.max(idx[0])+PAD_SIZE]

    points = [top, left, right, bottom]
    points = np.array(points).astype(np.int)

    return points



with torch.no_grad():
    model = maskRCNN_model()
    for path, dirs, files in os.walk("./ims/"):
      for filename in files:
        #extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)
        #extreme_points_ori = np.array(bbox).astype(np.int)
        image_path = path + "/" + filename
        image = np.array(Image.open(image_path))

        # Get the mask for person from maskRCNN and compute the extreme points using the mask
        _, _, mask, _ = get_maskRCNN_predictions(model, image_path)
        extreme_points_ori = get_EP_by_mask(mask)

        #extreme_points_ori = get_extreme_points(BBox)
        #extreme_points_ori = np.array([[205,60],[3,450],[275,475],[560,470]]).astype(np.int)
                                        

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        #results.append(result)
        results = result

        # Plot the results
        #plt.imshow(//helpers.overlay_masks(image / 255, results))
        #plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
        out_img = helpers.overlay_masks(image / 255, results)
        mplimg.imsave("./output/output_" + filename, out_img)
