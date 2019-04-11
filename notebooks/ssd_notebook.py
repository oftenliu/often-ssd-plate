import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization




def traverse(f,imageset):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            ext = os.path.splitext(tmp_path)[-1][1:]
            if ext == "jpg" or ext == "png" or ext == "bmp":
                #print('文件: %s'%tmp_path)
                imageset.append(tmp_path)
        else:
            print('文件夹：%s'%tmp_path)
            traverse(tmp_path,imageset)



# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, end_points = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../log4/model.ckpt-130001'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.85, nms_threshold=0.85, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rend_points,rbbox_img = isess.run([image_4d, predictions, localisations, end_points,bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
    
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes,rend_points

path = '../noplate/'
savepath = '../res/'
dataset = []
traverse(path,dataset)
image_size = len(dataset)


for index in range(0,image_size):
    print(dataset[index])
    img = cv2.imread(dataset[index])
    # file = "../nores/plate3531.jpg"
    # img = cv2.imread(file)
    rclasses, rscores, rbboxes,rend_points =  process_image(img)

    # for i, layer in enumerate(rend_points):
    #     visualize_utils.create_dir(dir_prefix + layer)
    #     for j in range(conv_out[i].shape[3]):
    #         visualize.plot_conv_output(conv_out[i], dir_prefix + layer, str(j), filters_all=False, filters=[j])




    visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    iname = dataset[index].rsplit('/', 1)[-1]
    file = savepath + iname
    cv2.imwrite(file,img)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        