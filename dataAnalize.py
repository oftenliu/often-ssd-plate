# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#　分析原始数据的尺度分布
#
#
# ==============================================================================
"""Generic evaluation script that evaluates a SSD model
on a given dataset."""
import math
import sys
import six
import time
import cv2
import numpy as np
import tensorflow as tf
import tf_extended as tfe
import tf_utils
from tensorflow.python.framework import ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
import matplotlib.pyplot as plt
# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_boolean(
    'wait_for_checkpoints', False, 'Wait for new checkpoints in the eval loop.')


FLAGS = tf.app.flags.FLAGS


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        # =================================================================== #
        # Dataset + SSD model + Pre-processing
        # =================================================================== #
        print(FLAGS.dataset_split_name)
        print(FLAGS.batch_size)
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        print(dataset.num_samples)
        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)

        # Evaluation shape and associated anchors: eval_image_size
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.eval_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== # 
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    common_queue_capacity=2 * FLAGS.batch_size,
                    common_queue_min=FLAGS.batch_size,
                    shuffle=False)
            # Get for SSD network: image, labels, bboxes.
            [gimage, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            if FLAGS.remove_difficult:
                [gdifficults] = provider.get(['object/difficult'])
            else:
                gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)

            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes, gbbox_img = \
                image_preprocessing_fn(gimage, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)

            # Encode groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = \
                ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] * 5 + [len(ssd_anchors)] * 3

            # Evaluation batch.
            r = tf.train.batch(
                tf_utils.reshape_list([image, glabels, gbboxes, gdifficults, gbbox_img,
                                       gclasses, glocalisations, gscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size,
                dynamic_pad=True)

            (b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses,
             b_glocalisations, b_gscores) = tf_utils.reshape_list(r, batch_shape)
        with tf.Session() as sess:
            ini_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(ini_op)
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess,coord=coord)
            plate_sizes = np.zeros((6,), dtype = np.int64) 
            plate_ratios = []
            widths = []
            heights = []
            for i in range(209):#(209053):
                image,gscores,gbboxes = sess.run([b_image,b_gscores,b_gbboxes]) #输出第一维是输出层的数量:6 第二维是batch_size
                # print(g_shape)
                # print(ori_image.shape)
                # cv2.imshow("origal",ori_image)
                # cv2.waitKey(0)
                # cv2.imshow("preimage",preimage)
                # cv2.waitKey(0)                
                for sample_index in range(len(gbboxes)):
                    sample = gbboxes[sample_index]
                    for object_index in range(len(sample)):
                        ymin = sample[object_index][0]
                        xmin = sample[object_index][1]
                        ymax = sample[object_index][2]
                        xmax = sample[object_index][3]
                        width = xmax - xmin
                       
                        height = ymax -ymin
                        #print(width,height)

                        plate_ratios.append(width/height)
                        widths.append(width)
                        heights.append(height)
                        p1 = (int(xmin * 300), int(ymin * 300) )
                        p2 = (int(xmax * 300), int(ymax * 300))
                        #print(p1,p2)
                        cv2.rectangle(image[sample_index], p1, p2, [0,255,255], 1)
                    # cv2.imshow(str(sample_index),image[sample_index])
                    # cv2.waitKey(0)



                inputlayer_size = len(gscores)                                
                layers_samples_max = []
                for inputlayer_index in range(0,inputlayer_size):

                    sample_size = len(gscores[inputlayer_index])
                    sample_score = gscores[inputlayer_index]
                    layer_samples_max = []
                    for sample_index in range(0,sample_size):
                        layer_samples_max.append( np.max(sample_score[sample_index]) )
                        #print("the layer "+ str(inputlayer_index)+" the sample " + str(sample_index) + " the max value is " +str(np.max(sample_score[sample_index])))
                    layers_samples_max.append(layer_samples_max)      
                layers_samples_max = np.asarray(layers_samples_max).swapaxes(0,1)
                sample_num = len(layers_samples_max)

                for sample_index in range(sample_num):
                    max_value = np.max(layers_samples_max[sample_index])
                    max_index = np.where(layers_samples_max[sample_index]==max_value)
                    #print(max_index)
                    if max_value == 0:
                        continue
                    
                    index_size =  len(max_index)
                    for index in range(index_size):
                        value = max_index[index]
                        #print(value)
                        plate_sizes[value] = plate_sizes[value] + 1
                #print(layers_samples_max)
            #print(gscores)
            min_ratio = np.nanmin(np.asarray(plate_ratios))
            max_ratio = np.nanmax(np.asarray(plate_ratios))
            print(min_ratio,max_ratio)
            da = np.arange(min_ratio, max_ratio, 0.1) 
            print(da.size) 

            fig = plt.figure(22)
            ax = plt.subplot(221)          
            plt.hist(plate_ratios,da.size)
            ax.set_title("bins=%d,title=ratio(width/height)"%da.size)


            min_width = np.min(np.asarray(widths))
            max_width = np.max(np.asarray(widths))
            da = np.arange(min_width, max_width, 0.1)   
            ax = plt.subplot(222)              
            plt.hist(np.asarray(widths),da)
            ax.set_title("bins=%d,title=width"%da.size)

            min_height = np.min(np.asarray(heights))
            max_height = np.max(np.asarray(heights))
            da = np.arange(min_height, max_height, 0.1)      
            ax = plt.subplot(223)        
            plt.hist(np.asarray(heights),da)            
            ax.set_title("bins=%d,title=height"%da.size)


            da = np.arange(0, len(plate_sizes), 1)   
            ax = plt.subplot(224)    
            plt.plot(da, plate_sizes, 's')
            ax.set_title("title=sizes")
            fig.tight_layout()        
            plt.show()            
                
            plt.savefig("./4.png")

            
            coord.request_stop()

            coord.join(thread)

if __name__ == '__main__':
    tf.app.run()
