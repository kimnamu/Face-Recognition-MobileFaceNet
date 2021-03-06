# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data_process import load_data
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import re
import os
import cv2

from mtcnn import MTCNN
import glob
import time

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def preprocessing(image, detector, filename = ""):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = detector.detect_faces(img)
    if detection.__len__() == 0: 
        print("fail to detect face")
        return [], False
    x, y, w, h = detection[0]['box']
    img = img[y:y+h, x:x+w]
    if img.size == 0: 
        print("fail to detection correctly")
        return [], False
    img = cv2.resize(img, (112,112))
    if filename is not "":
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = img - 127.5
    img = img * 0.0078125
    return img, True

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            load_model(args.model)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # face detection
            detector = MTCNN()

            img = cv2.imread("registration/003/002.jpg")
            img, flag = preprocessing(img, detector, "image1.jpg")
            if flag is False : 
                pass
            feed_dict = {inputs_placeholder: [img]}
            embed1 = sess.run(embeddings, feed_dict=feed_dict)

            img = cv2.imread("registration/003/001.jpg")
            img, flag = preprocessing(img, detector, "image2.jpg")
            if flag is False : 
                pass
            feed_dict = {inputs_placeholder: [img]}
            embed2 = sess.run(embeddings, feed_dict=feed_dict)

            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

            if dist < 0.95: # 1.19:
                print("Same", dist)
            else:
                print("Not same", dist)
                        

def parse_arguments(argv):
    '''test parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./arch/pretrained_model')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_vgg_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



# diff = np.subtract(embed1, embed2)
# dist = np.sum(np.square(diff), 1)

# Runnning forward pass on lfw images
# thresholds max: 1.25 <=> min: 1.19
# total time 218.712s to evaluate 12000 images of lfw
# Accuracy: 0.994+-0.004
# Validation rate: 0.98367+-0.00924 @ FAR=0.00133
# fpr and tpr: 0.503 0.872
# Area Under Curve (AUC): 0.999
# Equal Error Rate (EER): 0.007


# detector.detect_faces(img)
# [
#     {
#         'box': [277, 90, 48, 63], # [x, y, width, height]
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]
