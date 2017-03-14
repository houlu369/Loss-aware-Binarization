from __future__ import print_function

import sys
import os
import time
import ipdb
import numpy as np
np.random.seed(1234)  
from argparse import ArgumentParser

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
import lab
import optimizer

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict


def main(method,LR_start,Binarize_weight_only):
	
	# BN parameters
	name = "mnist"
	print("dataset = "+str(name))

	print("Binarize_weight_only="+str(Binarize_weight_only))

	print("Method = "+str(method))

	# alpha is the exponential moving average factor
	alpha = .1
	print("alpha = "+str(alpha))
	epsilon = 1e-4
	print("epsilon = "+str(epsilon))
	
	batch_size = 100
	print("batch_size = "+str(batch_size))

	num_epochs = 50
	print("num_epochs = "+str(num_epochs))

	# network structure
	num_units = 2048
	print("num_units = "+str(num_units))
	n_hidden_layers = 3
	print("n_hidden_layers = "+str(n_hidden_layers))

	print("LR_start = "+str(LR_start))
	LR_decay = 0.1
	print("LR_decay="+str(LR_decay))

	if Binarize_weight_only =="w":
		activation = lasagne.nonlinearities.rectify
	else:
		activation = lab.binary_tanh_unit
	print("activation = "+ str(activation))


	print('Loading MNIST dataset...')
	
	train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
	valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
	test_set = MNIST(which_set= 'test', center = True)
	
	# bc01 format
	train_set.X = train_set.X.reshape(-1, 1, 28, 28)
	valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
	test_set.X = test_set.X.reshape(-1, 1, 28, 28)
	
	# flatten targets
	train_set.y = np.hstack(train_set.y)
	valid_set.y = np.hstack(valid_set.y)
	test_set.y = np.hstack(test_set.y)
	
	# Onehot the targets
	train_set.y = np.float32(np.eye(10)[train_set.y])    
	valid_set.y = np.float32(np.eye(10)[valid_set.y])
	test_set.y = np.float32(np.eye(10)[test_set.y])
	
	# for hinge loss
	train_set.y = 2* train_set.y - 1.
	valid_set.y = 2* valid_set.y - 1.
	test_set.y = 2* test_set.y - 1.

	print('Building the MLP...') 
	
	# Prepare Theano variables for inputs and targets
	input = T.tensor4('inputs')
	target = T.matrix('targets')
	LR = T.scalar('LR', dtype=theano.config.floatX)

	mlp = lasagne.layers.InputLayer(
			shape=(None, 1, 28, 28),
			input_var=input)
	
	for k in range(n_hidden_layers):
		mlp = lab.DenseLayer(
				mlp, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=num_units,
				method = method)                  	
		mlp = batch_norm.BatchNormLayer(
				mlp,
				epsilon=epsilon, 
				alpha=alpha)
		mlp = lasagne.layers.NonlinearityLayer(
				mlp,
				nonlinearity = activation)

	mlp = lab.DenseLayer(
				mlp, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=10,
				method = method)      
				  
	mlp = batch_norm.BatchNormLayer(
			mlp,
			epsilon=epsilon, 
			alpha=alpha)

	train_output = lasagne.layers.get_output(mlp, deterministic=False)
	
	# squared hinge loss
	loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
	
	if method!="FPN":
		
		# W updates
		W = lasagne.layers.get_all_params(mlp, binary=True)
		W_grads = lab.compute_grads(loss,mlp)
		updates = optimizer.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
		updates = lab.clipping_scaling(updates,mlp)
		
		# other parameters updates
		params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
		updates = OrderedDict(updates.items() + optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

		## update 2nd moment, can get from the adam optimizer also
		updates3 = OrderedDict()
		acc_tag = lasagne.layers.get_all_params(mlp, acc=True)	
		idx = 0
		beta2 = 0.999
		for acc_tag_temp in acc_tag:
			updates3[acc_tag_temp]= acc_tag_temp*beta2 + W_grads[idx]*W_grads[idx]*(1-beta2)
			idx = idx+1

		updates = OrderedDict(updates.items() + updates3.items())

	else:
		params = lasagne.layers.get_all_params(mlp, trainable=True)
		updates = optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR)

	test_output = lasagne.layers.get_output(mlp, deterministic=True)
	test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
	test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
	
	# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
	# and returning the corresponding training loss:
	train_fn = theano.function([input, target, LR], loss, updates=updates)
	val_fn = theano.function([input, target], [test_loss, test_err])

	print('Training...')
	
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
				default=0.01, help="Learning rate") 
	parser.add_argument("--w", type=str, dest="Binarize_weight_only",
				default="w", help="true:only binzrize w, false: binarize w and a")
	args = parser.parse_args()

	main(**vars(args))