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

        saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.compat.v1.get_default_session(), os.path.join(model_exp, ckpt_file))


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

def preprocessing(image, detector, filesave = ""):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = detector.detect_faces(img)
    if detection.__len__() == 0: 
        print("fail to detect face")
        return [], False, []
    x, y, w, h = detection[0]['box']
    img = img[y:y+h, x:x+w]
    if img.size == 0: 
        print("fail to detection correctly")
        return [], False, []
    img = cv2.resize(img, (112,112))
    if filesave is not "":
        cv2.imwrite(filesave, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = img - 127.5
    img = img * 0.0078125
    bbox = [x, y, w, h]
    return img, True, bbox

def load_pb(graph, path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    return graph

def registration(img, embeds_reg, detector, sess, inputs_placeholder, embeddings):
    dir_reg = "./registration"
    id_list = [os.path.join(dir_reg, d) for d in os.listdir(dir_reg) if os.path.isdir(os.path.join(dir_reg, d))]
    if len(id_list)==0:
        id_new = 1
    else:
        id_list.sort()
        id_new = int(id_list[-1].split('/')[2])+1
    os.mkdir("{}/{:03}".format(dir_reg, id_new))
    cv2.imwrite("{}/{:03}/001.jpg".format(dir_reg, id_new), img)

    file = "{}/{:03}/001.jpg".format(dir_reg, id_new)
    id_embed = file.split('/')[2]
    embedFile = (file[0:-3]+"npy")
    
    img, flag, bbox = preprocessing(img, detector)
    if flag is False :
        return embeds_reg, flag

    feed_dict = {inputs_placeholder: [img]}
    embed = sess.run(embeddings, feed_dict=feed_dict)
    np.save(embedFile, embed)
    embeds_reg[id_embed] = embed

    return embeds_reg, flag



def main(args):
    # path_to_pb = "./arch/pretrained2/new_model.pb"
    # path_to_pb = "./arch/pretrained_model/MobileFaceNet_TF.pb"

    # with tf.io.gfile.GFile(path_to_pb, "rb") as f:
    #     graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        # tf.import_graph_def(graph_def, name='')
        with tf.compat.v1.Session() as sess:
            cap = cv2.VideoCapture(0)
            # 1. ckpt
            # Load the model
            load_model(args.model)
            
            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")

            # 2. (temp) save pb model
            # tf.io.write_graph(sess.graph_def, '.', './arch/pretrained2/new_model.pb', as_text=False)

            # 2. pb version
            # graph = load_pb(graph, "./arch/pretrained2/new_model.pb")
            # path_to_pb = "./arch/pretrained2/new_model.pb"
            # with tf.gfile.GFile(path_to_pb, "rb") as f: # pb
            #     graph_def = tf.GraphDef()
            #     graph_def.ParseFromString(f.read())
            # tf.import_graph_def(graph_def, name='') # pb
            # inputs_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("img_inputs:0")
            # embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            inputs_placeholder = graph.get_tensor_by_name('img_inputs:0')
            embeddings = graph.get_tensor_by_name('embeddings:0')

            # face detection
            detector = MTCNN()

            # Embedding Registered Images
            dir_reg = "./registration"
            if os.path.exists(dir_reg):
                filelist = glob.glob("./registration/*/*.jpg") # './registration/001/001.jpg' 
            else:
                os.mkdir(dir_reg)
                filelist = []
            filelist.sort()
            embeds_reg = {}
            for file in filelist:
                id_embed = file.split('/')[2]
                embedFile = (file[0:-3]+"npy")
                if os.path.exists(embedFile):
                    embed = np.load(embedFile)
                    if id_embed in embeds_reg:
                        embeds_reg[id_embed] = np.append(embeds_reg[id_embed], embed, axis = 0)
                    else:
                        embeds_reg[id_embed] = embed
                    continue
                img = cv2.imread(file)
                img, flag, bbox = preprocessing(img, detector)
                if flag is False : continue
                feed_dict = {inputs_placeholder: [img]}
                embed = sess.run(embeddings, feed_dict=feed_dict)
                np.save(embedFile, embed)
                if id_embed in embeds_reg:
                    embeds_reg[id_embed] = np.append(embeds_reg[id_embed], embed, axis = 0)
                else:
                    embeds_reg[id_embed] = embed

            while(True):
                # Read Camera Image
                ret, cam_img = cap.read()
                if ret == False: break
                # cam_img = cv2.imread(filelist[0])
                start = time.time()
                img, flag, bbox = preprocessing(cam_img, detector, "image2.jpg")
                feed_dict = {inputs_placeholder: [img]}
                if flag is False : continue

                key = cv2.waitKey(25)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                if key != -1 and chr(key) == 'r':
                    embeds_reg, flag = registration(cam_img, embeds_reg, detector, sess, inputs_placeholder, embeddings)
                    print("Registration is {}".format("success" if flag else "fail"))
                    continue

                embed_cmp = sess.run(embeddings, feed_dict=feed_dict)
                min_dist = 10
                for key in embeds_reg:
                    for i, embed_reg in enumerate(embeds_reg[key]):
                        diff = np.subtract(embed_reg, embed_cmp)
                        dist = np.sum(np.square(diff), 1)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = key
                            img_name ="./registration/{}/{:03}.jpg".format(key, i+1)
                if min_dist != 10:
                    img = cv2.imread(img_name)
                    cv2.imwrite("./image1.jpg", img)
                
                end = time.time()

                name = "Unregistered"
                if min_dist < 0.95: # 1.19:
                    name = str("ID:{}".format(min_id))
                    print("{}(Score:{:.3})".format(name, float(min_dist)))
                else:
                    img = cv2.imread("./registration/unregistered.jpg")
                    cv2.imwrite("./image1.jpg", img)
                    print("{}(Score:{:.3})".format(name, float(min_dist)))                   

                x,y,w,h = bbox
                image = cv2.rectangle(cam_img, (x, y), (x + w, y + h), (36,255,12), 1)
                cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.imshow("Face Recognition", image)
                print("Time elapsed during the calculation: {:.3} sec, {:.3} fps\n".format(end - start, 1.0/(end-start)))
                
                        

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


## This is the result of runnning forward pass on lfw images
# thresholds max: 1.25 <=> min: 1.19
# total time 218.712s to evaluate 12000 images of lfw
# Accuracy: 0.994+-0.004
# Validation rate: 0.98367+-0.00924 @ FAR=0.00133
# fpr and tpr: 0.503 0.872
# Area Under Curve (AUC): 0.999
# Equal Error Rate (EER): 0.007


## This is the example of MTCNN detection result
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


"**((*"

