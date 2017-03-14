from __future__ import print_function

import sys
import os
import time
import ipdb

import numpy as np
np.random.seed(1234) 

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
import lab
import optimizer

from pylearn2.datasets.svhn import SVHN
from pylearn2.utils import serial

from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.utils import serial

from collections import OrderedDict
from argparse import ArgumentParser

def main(method,LR_start,Binarize_weight_only):
	
	name = "svhn"
	print("dataset = "+str(name))

	print("Binarize_weight_only="+str(Binarize_weight_only))

	print("Method = "+str(method))

	# alpha is the exponential moving average factor
	alpha = .1
	print("alpha = "+str(alpha))
	epsilon = 1e-4
	print("epsilon = "+str(epsilon))
	
	# Training parameters
	batch_size = 50
	print("batch_size = "+str(batch_size))
	
	num_epochs = 50
	print("num_epochs = "+str(num_epochs))

	print("LR_start = "+str(LR_start))
	LR_decay = 0.1
	print("LR_decay="+str(LR_decay))
	# BTW, LR decay might good for the BN moving average...

	if Binarize_weight_only =="w":
		activation = lasagne.nonlinearities.rectify
	else:
		activation = lab.binary_tanh_unit
	print("activation = "+ str(activation))

	## number of filters in the first convolutional layer
	K = 64 
	print("K="+str(K))

	print('Building the CNN...') 
	
	# Prepare Theano variables for inputs and targets
	input = T.tensor4('inputs')
	target = T.matrix('targets')
	LR = T.scalar('LR', dtype=theano.config.floatX)

	l_in = lasagne.layers.InputLayer(
			shape=(None, 3, 32, 32),
			input_var=input)
	
	# 128C3-128C3-P2             
	l_cnn1 = lab.Conv2DLayer(
			l_in, 
			num_filters=K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_bn1 = batch_norm.BatchNormLayer(
			l_cnn1,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl1 = lasagne.layers.NonlinearityLayer(
			l_bn1,
			nonlinearity = activation)

	l_cnn2 = lab.Conv2DLayer(
			l_nl1, 
			num_filters=K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_mp1 = lasagne.layers.MaxPool2DLayer(l_cnn2, pool_size=(2, 2))
	
	l_bn2 = batch_norm.BatchNormLayer(
			l_mp1,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl2 = lasagne.layers.NonlinearityLayer(
			l_bn2,
			nonlinearity = activation)			
	# 256C3-256C3-P2             
	l_cnn3 = lab.Conv2DLayer(
			l_nl2, 
			num_filters=2*K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_bn3 = batch_norm.BatchNormLayer(
			l_cnn3,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl3 = lasagne.layers.NonlinearityLayer(
			l_bn3,
			nonlinearity = activation)
			
	l_cnn4 = lab.Conv2DLayer(
			l_nl3, 
			num_filters=2*K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_mp2 = lasagne.layers.MaxPool2DLayer(l_cnn4, pool_size=(2, 2))
	
	l_bn4 = batch_norm.BatchNormLayer(
			l_mp2,
			epsilon=epsilon, 
			alpha=alpha)
	
	l_nl4 = lasagne.layers.NonlinearityLayer(
			l_bn4,
			nonlinearity = activation)

	# 512C3-512C3-P2              
	l_cnn5 = lab.Conv2DLayer(
			l_nl4, 
			num_filters=4*K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_bn5 = batch_norm.BatchNormLayer(
			l_cnn5,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl5 = lasagne.layers.NonlinearityLayer(
			l_bn5,
			nonlinearity = activation)
				  
	l_cnn6 = lab.Conv2DLayer(
			l_nl5, 
			num_filters=4*K, 
			filter_size=(3, 3),
			pad=1,
			nonlinearity=lasagne.nonlinearities.identity,
			method = method)
	
	l_mp3 = lasagne.layers.MaxPool2DLayer(l_cnn6, pool_size=(2, 2))
	
	l_bn6 = batch_norm.BatchNormLayer(
			l_mp3,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl6 = lasagne.layers.NonlinearityLayer(
			l_bn6,
			nonlinearity = activation)

	# print(cnn.output_shape)
	
	# 1024FP-1024FP-10FP            
	l_dn1 = lab.DenseLayer(
				l_nl6, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=1024,
				method = method)      
				  
	l_bn7 = batch_norm.BatchNormLayer(
			l_dn1,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl7 = lasagne.layers.NonlinearityLayer(
			l_bn7,
			nonlinearity = activation)

	l_dn2 = lab.DenseLayer(
				l_nl7, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=1024,
				method = method)      
				  
	l_bn8 = batch_norm.BatchNormLayer(
			l_dn2,
			epsilon=epsilon, 
			alpha=alpha)

	l_nl8 = lasagne.layers.NonlinearityLayer(
			l_bn8,
			nonlinearity = activation)

	l_dn3 = lab.DenseLayer(
				l_nl8, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=10,
				method = method)      
				  
	l_out = batch_norm.BatchNormLayer(
			l_dn3,
			epsilon=epsilon, 
			alpha=alpha)

	train_output = lasagne.layers.get_output(l_out, deterministic=False)
	
	
	# squared hinge loss
	loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
	
	if method!="FPN":
		# W updates
		W = lasagne.layers.get_all_params(l_out, binary=True)
		W_grads = lab.compute_grads(loss,l_out)
		updates = optimizer.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
		updates = lab.clipping_scaling(updates,l_out)
		
		# other parameters updates
		params = lasagne.layers.get_all_params(l_out, trainable=True, binary=False)
		updates = OrderedDict(updates.items() + optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

		## update 2nd moment, can get from the adam optimizer also
		updates3 = OrderedDict()
		acc_tag = lasagne.layers.get_all_params(l_out, acc=True)	
		idx = 0
		beta2 = 0.999   
		for acc_tag_temp in acc_tag:
			updates3[acc_tag_temp]= acc_tag_temp*beta2 + W_grads[idx]*W_grads[idx]*(1-beta2)
			idx = idx+1

		updates = OrderedDict(updates.items() +  updates3.items())	
	else:
		params = lasagne.layers.get_all_params(l_out, trainable=True)
		updates = optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR)

	test_output = lasagne.layers.get_output(l_out, deterministic=True)
	test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
	test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
	
	# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
	# and returning the corresponding training loss:
	train_fn = theano.function([input, target, LR], loss, updates=updates)
	val_fn = theano.function([input, target], [test_loss, test_err])


	## load data
	print('Loading SVHN dataset')
	
	train_set = SVHN(
			which_set= 'splitted_train',
			# which_set= 'valid',
			path= "${SVHN_LOCAL_PATH}",
			axes= ['b', 'c', 0, 1])
	 
	valid_set = SVHN(
		which_set= 'valid',
		path= "${SVHN_LOCAL_PATH}",
		axes= ['b', 'c', 0, 1])
	
	test_set = SVHN(
		which_set= 'test',
		path= "${SVHN_LOCAL_PATH}",
		axes= ['b', 'c', 0, 1])
	
	# bc01 format
	# print train_set.X.shape
	train_set.X = np.reshape(train_set.X,(-1,3,32,32))
	valid_set.X = np.reshape(valid_set.X,(-1,3,32,32))
	test_set.X = np.reshape(test_set.X,(-1,3,32,32))

	train_set.y = np.array(train_set.y).flatten()
	valid_set.y = np.array(valid_set.y).flatten()
	test_set.y = np.array(test_set.y).flatten()

	# Onehot the targets
	train_set.y = np.float32(np.eye(10)[train_set.y])    
	valid_set.y = np.float32(np.eye(10)[valid_set.y])
	test_set.y = np.float32(np.eye(10)[test_set.y])
	
	# for hinge loss
	train_set.y = 2* train_set.y - 1.
	valid_set.y = 2* valid_set.y - 1.
	test_set.y = 2* test_set.y - 1.    


	print('Training...')

	# ipdb.set_trace()
	lab.train(
			name, method,
			train_fn,val_fn,
			batch_size,
			LR_start,LR_decay,
			num_epochs,
			train_set.X,train_set.y,
			valid_set.X,valid_set.y,
			test_set.X,test_set.y)
	

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--method", type=str, dest="method",
				default="LAB", help="Method used")
	parser.add_argument("--lr_start",  type=float, dest="LR_start",
				default=0.001, help="Learning rate") 
	parser.add_argument("--w", type=str, dest="Binarize_weight_only",
				default="w", help="true:only binzrize w, false: binarize w and a")
	args = parser.parse_args()

	main(**vars(args))