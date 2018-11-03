#!/usr/bin/env python
"""
PEPPER ROBOT - Implementation of N2NMN model (https://github.com/ronghanghu/n2nmn)

This code implements the visual question answering model from (R. Hu, J. Andreas et al., 2017) into Pepper robot. The model was trained on CLEVR dataset and is thus able to recognize object primitives, their attributes and spaital relations.

Date: 4.5.2018
Author: Gabriela Sejnova <gabriela.sejnova@cvut.cz>
Copyright (c) CIIRC CTU in Prague  - All Rights Reserved
"""

from __future__ import absolute_import, division, print_function

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()
gpu_id = args.gpu_id  # set GPU id to use
import os;

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
from robot import Pepper

sys.path.append('./exp_clevr/data/')
sys.path.append('./')
import numpy as np
from numpy import array
import text_processing
import tensorflow as tf
from nmn3_assembler import Assembler
from nmn3_model import NMN3Model
from models_clevr import vgg_net
from glob import glob
import time, paramiko, os, cv2
import csv
from collections import defaultdict
import json 

""" Parameters """
# Image
H = 320
W = 480
H_feat = 10
W_feat = 15
D_feat = 512

# NN module parameters
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 512
num_layers = 2
T_encoder = 45
T_decoder = 20
prune_filter_module = True
vgg_net_model = './exp_clevr/tfmodel/vgg_net/vgg_net.tfmodel'

img_folder = "/home/popelka/CLEVR_generator/output/images/"
questions = ["How many objects are there", "How many spheres are there", "How many cubes are there", "How many cylinders are there","How many rubber objects are there", "How many metal objects are there","How many red objects are there","How many green objects are there","How many blue objects are there","How many yellow objects are there","How many gray objects are there","How many brown objects are there","How many purple objects are there","How many cyan objects are there","How many large objects are there", "How many small objects are there"]
csvfile = "/home/popelka/vqa_clevr/exp_clevr/counting_tokens.csv"

class PepperDescribeScn(object):

	def __init__(self):
		for root,dirs,files in os.walk(img_folder):
			''' Run model on images in folder'''
			files = sorted(files)
			for filename in files:
				spheres=cubes=cylinders=metal=rubber=0
				red=grn=blu=yel=gra=bro=pur=cya=lar=sma=0
				hits = []
				img = img_folder + filename
				print(img)
				self.tokens = []
				bgr_img = cv2.imread(img)
				b,g,r = cv2.split(bgr_img)       # get b,g,r
				rgb_img = cv2.merge([r,g,b])   
				self.scene_img = rgb_img
				self.feature_extraction()
				for question in questions:
					print(question)
				     	vqa = VQA(question, self.scene_img, self.pool5_val)
					vqa.load_nmn3_model()
					self.get_answer(vqa)
					self.token = self.pepper_tokens
					self.tokens.append(self.token)
				scenename = os.path.splitext(filename)[0]     
				
				''' Get ground truth from .json'''
				with open("/home/popelka/CLEVR_generator/output/scenes/" + scenename + ".json", 'r') as f:
					ground_t = json.load(f)
					for n in range(len(ground_t['objects'])):
						objects = ground_t['objects'] 
						if ground_t['objects'][n]['shape'] == "sphere":
							spheres = spheres + 1
						elif ground_t['objects'][n]['shape'] == "cube":
							cubes = cubes + 1
						elif ground_t['objects'][n]['shape'] == "cylinder":
							cylinders = cylinders + 1
						if ground_t['objects'][n]['material'] == "rubber":
							rubber = rubber + 1
						elif ground_t['objects'][n]['material'] == "metal":
							metal = metal + 1
						if ground_t['objects'][n]['color'] == "red":
							red = red + 1
						elif ground_t['objects'][n]['color'] == "green":
							grn = grn + 1
						elif ground_t['objects'][n]['color'] == "blue":
							blu = blu + 1
						elif ground_t['objects'][n]['color'] == "yellow":
							yel = yel + 1
						elif ground_t['objects'][n]['color'] == "gray":
							gra = gra + 1
						elif ground_t['objects'][n]['color'] == "brown":
							bro = bro + 1
						elif ground_t['objects'][n]['color'] == "purple":
							pur = pur + 1
						elif ground_t['objects'][n]['color'] == "cyan":
							cya = cya + 1
						if ground_t['objects'][n]['size'] == "large":
							lar = lar + 1
						elif ground_t['objects'][n]['size'] == "small":
							sma = sma + 1

				#ground_truth = [len(objects), spheres, cubes, cylinders, rubber, metal, red, grn, blu, yel, gra, bro, pur, cya, lar, sma]
				a = [1,8]
				ground_truth = [[1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8]]

				print("ground truth", ground_truth)
				print("tokens",self.tokens)
				''' Get number of correct answers'''
				for x in range(len(ground_truth)):
					if set(ground_truth[x]) == set(self.tokens[x]):
						hits.append(1)
					elif set(ground_truth[x]) == set(self.tokens[x]):
						hits.append(0)
				print("hits", hits)

				'''Export to .csv'''
				data = [[filename] + [""] + hits]
				with open(csvfile, "ab") as output:
					writer = csv.writer(output, lineterminator='\n')
			    		writer.writerows(data)      
				data = []


	def get_answer(self, vqa):
        	self.pepper_tokens = vqa.run_test()
		
	def feature_extraction(self):
		""" Extracts visual features from image via VGG neural network """
		tf.reset_default_graph()
		self.image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
		self.pool5 = vgg_net.vgg_pool5(self.image_batch, 'vgg_net')
		self.saver = tf.train.Saver()
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		self.saver.restore(self.sess, vgg_net_model)
		# im = skimage.io.imread(image_basedir)[..., :3]
		# im = skimage.io.scene_img[..., :3]
		self.im_val = (self.scene_img[np.newaxis, ...] - vgg_net.channel_mean)
		self.pool5_val = self.pool5.eval({self.image_batch: self.im_val}, self.sess)
		self.sess.close()
		tf.reset_default_graph()

		return self.pool5_val

class VQA(object):

    def __init__(self, question, scene_img, pool5_val):
        """ Data files """
        self.question = question
        self.scene_img = scene_img
        self.snapshot_file = './exp_clevr/tfmodel/clevr_rl_gt_layout/00050000'
        self.vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
        self.vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
        self.vocab_answer_file = './exp_clevr/data/answers_clevr.txt'
	self.pool5_val = pool5_val

    def load_nmn3_model(self):
        """Initialize model session"""
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True),
            allow_soft_placement=False, log_device_placement=False))

        self.raw_question = self.question
        self.question_tokens = text_processing.tokenize(self.raw_question)
        self.assembler = Assembler(self.vocab_layout_file)
        self.vocab_dict = text_processing.VocabDict(self.vocab_question_file)
        self.answer_dict = text_processing.VocabDict(self.vocab_answer_file)

        self.num_vocab_txt = self.vocab_dict.num_vocab
        self.num_vocab_nmn = len(self.assembler.module_names)
        self.num_choices = self.answer_dict.num_vocab

        """Network inputs - placeholders"""
        self.input_seq_batch = tf.placeholder(tf.int32, [None, None])
        self.seq_length_batch = tf.placeholder(tf.int32, [None])
        self.image_feat_batch = tf.placeholder(tf.float32, [None, H_feat, W_feat, D_feat])
        self.expr_validity_batch = tf.placeholder(tf.bool, [None])

        """The model for testing"""
        self.nmn3_model_tst = NMN3Model(
            self.image_feat_batch, self.input_seq_batch,
            self.seq_length_batch, T_decoder=T_decoder,
            num_vocab_txt=self.num_vocab_txt, embed_dim_txt=embed_dim_txt,
            num_vocab_nmn=self.num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
            lstm_dim=lstm_dim, num_layers=num_layers,
            assembler=self.assembler,
            encoder_dropout=False,
            decoder_dropout=False,
            decoder_sampling=False,
            num_choices=self.num_choices)

        self.snapshot_saver = tf.train.Saver(max_to_keep=None)
        self.snapshot_saver.restore(self.sess, self.snapshot_file)
        self.input_seq = np.zeros((T_encoder, 1), np.int32)
        self.seq_length = np.zeros(1, np.int32)
        self.image_feat = np.zeros((1, H_feat, W_feat, D_feat), np.float32)
        self.image_feat = self.pool5_val

    def run_test(self):
        print('Running test...')
        self.layout_valid = 0
        self.answer_word_list = self.answer_dict.word_list
        self.output_answers = []
        self.question_inds = [self.vocab_dict.word2idx(w) for w in self.question_tokens]
        self.seq_length = len(self.question_inds)

        self.input_seq[:self.seq_length, 0] = self.question_inds
        input_seq = self.input_seq
        seq_length = [self.seq_length]
        image_feat = self.image_feat
        self.h = self.sess.partial_run_setup(
            [self.nmn3_model_tst.predicted_tokens, self.nmn3_model_tst.scores],
            [self.input_seq_batch, self.seq_length_batch, self.image_feat_batch,
             self.nmn3_model_tst.compiler.loom_input_tensor, self.expr_validity_batch])

        """Part 0 & 1: Run Convnet and generate module layout"""
        self.tokens = self.sess.partial_run(self.h, self.nmn3_model_tst.predicted_tokens,
                                            feed_dict={self.input_seq_batch: input_seq,
                                                       self.seq_length_batch: seq_length,
                                                       self.image_feat_batch: image_feat})
	tokens = list(self.tokens[0]) + list(self.tokens[1])
	

        """ Assemble the layout tokens into network structure"""
        #expr_list, expr_validity_array = self.assembler.assemble(self.tokens)
        #self.layout_valid += np.sum(expr_validity_array)

        """Build TensorFlow Fold input for NMN"""
        #expr_feed = self.nmn3_model_tst.compiler.build_feed_dict(expr_list)
        #expr_feed[self.expr_validity_batch] = expr_validity_array

        """Part 2: Run NMN and learning steps"""
        #self.scores_val = self.sess.partial_run(self.h, self.nmn3_model_tst.scores, feed_dict=expr_feed)

        """Get answer"""
        #self.predictions = np.argmax(self.scores_val, axis=1)
        #self.output_answer = [self.answer_word_list[p] for p in self.predictions]
        self.sess.close()
        tf.reset_default_graph()
        #print("Answer:", self.output_answer[0])
        #self.answer = self.output_answer[0]
        #return self.answer
	return tokens

if __name__ == "__main__":
	

	PepperDescribeScn()

            
            
            
