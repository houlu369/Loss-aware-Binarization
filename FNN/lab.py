import time

from collections import OrderedDict

import numpy as np
np.random.seed(1234) 

import ipdb

import theano
import theano.tensor as T

import lasagne
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Round3(UnaryScalarOp):
	
	def c_code(self, node, name, (x,), (z,), sub):
		return "%(z)s = round(%(x)s);" % locals()
	
	def grad(self, inputs, gout):
		(gz,) = gout
		return gz, 
		
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
	return T.clip((x+1.)/2.,0,1)

def binary_tanh_unit(x):
	return 2.*round3(hard_sigmoid(x))-1.
	
def binary_sigmoid_unit(x):
	return round3(hard_sigmoid(x))
	

# The binarization function
def binarization(W,Wacc,method):
	
	if method == "FPN":
		Wb = W
	
	elif method == "LAB":
		L = (T.sqrt(Wacc) + 1e-8) 
		Wb = hard_sigmoid(W)
		Wb = round3(Wb)
		Wb = T.cast(T.switch(Wb,1.,-1.), theano.config.floatX) 

		alpha  = (T.abs_(L*W).sum()/L.sum()).astype('float32') 
		Wb = alpha*Wb		
								  
	return Wb


# This class extends the Lasagne DenseLayer to support LAB
class DenseLayer(lasagne.layers.DenseLayer):
	
	def __init__(self, incoming, num_units, method, **kwargs):
		
		self.method = method		
		num_inputs = int(np.prod(incoming.output_shape[1:]))
		g_init = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
		if self.method !="FPN":
			super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-g_init,g_init)), **kwargs)
			# add the binary tag to weights            
			self.params[self.W]=set(['binary'])
		else:
			super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
		# add the acc tag to 2nd momentum  
		self.acc_W = theano.shared(np.zeros((self.W.get_value(borrow=True)).shape, dtype='float32'))
		self.params[self.acc_W]=set(['acc'])

	def get_output_for(self, input, deterministic=False, **kwargs):
		
		self.Wb = binarization(self.W, self.acc_W, self.method)
		Wr = self.W
		self.W = self.Wb
			
		rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
		
		self.W = Wr
		
		return rvalue

# This class extends the Lasagne Conv2DLayer to support LAB
class Conv2DLayer(lasagne.layers.Conv2DLayer):
	
	def __init__(self, incoming, num_filters, filter_size, method,  **kwargs):
		
		self.method = method
		
		num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
		num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
		g_init = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))

		if self.method!="FPN":
			super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-g_init,g_init)), **kwargs)
			# add the binary tag to weights            
			self.params[self.W]=set(['binary'])
		else:
			super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
	
		self.acc_W = theano.shared(np.zeros((self.W.get_value(borrow=True)).shape, dtype='float32'))
		self.params[self.acc_W]=set(['acc'])


	def get_output_for(self, input, deterministic=False, **kwargs):
		
		self.Wb = binarization(self.W, self.acc_W, self.method)
		Wr = self.W
		self.W = self.Wb
		rvalue = super(Conv2DLayer, self).get_output_for(input, **kwargs)		
		self.W = Wr
		
		return rvalue

def compute_grads(loss,network):
		
	layers = lasagne.layers.get_all_layers(network)
	grads = []

	for layer in layers:	
		params = layer.get_params(binary=True)
		if params:
			grads.append(theano.grad(loss, wrt=layer.Wb))
				
	return grads

# This functions clips the weights after the parameter update 
def clipping_scaling(updates,network):
	
	layers = lasagne.layers.get_all_layers(network)
	updates = OrderedDict(updates)
	
	for layer in layers:	
		params = layer.get_params(binary=True)
		for param in params:   
			updates[param] = T.clip(updates[param],-1.,1.)
	return updates
		
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default train function in Lasagne yet)
def train(name,method,train_fn,val_fn,
			batch_size,
			LR_start,LR_decay,
			num_epochs,
			X_train,y_train,
			X_val,y_val,
			X_test,y_test):
	
	# This function trains the model a full epoch (on the whole dataset)
	def train_epoch(X,y,LR):
		
		loss = 0
		batches = len(X)/batch_size
		# move shuffle here to save memory
		shuffled_range = range(len(X))
		np.random.shuffle(shuffled_range)

		for i in range(batches):
			tmp_ind = shuffled_range[i*batch_size:(i+1)*batch_size]  
			newloss = train_fn(X[tmp_ind],y[tmp_ind],LR)      	
			loss +=newloss		
		loss/=batches
		
		return loss
	
	# This function tests the model a full epoch (on the whole dataset)
	def val_epoch(X,y):
		
		err = 0
		loss = 0
		batches = len(X)/batch_size
		
		for i in range(batches):
			new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
			err += new_err
			loss += new_loss
		
		err = err / batches * 100
		loss /= batches

		return err, loss
	

	best_val_err = 100
	best_epoch = 1
	LR = LR_start
	# We iterate over epochs:
	for epoch in range(1, num_epochs+1):
		
		start_time = time.time()
		train_loss = train_epoch(X_train,y_train,LR)
		
		val_err, val_loss = val_epoch(X_val,y_val)
		
		# test if validation error went down
		if val_err <= best_val_err:
			
			best_val_err = val_err
			best_epoch = epoch+1
			
			test_err, test_loss = val_epoch(X_test,y_test)
		
		epoch_duration = time.time() - start_time
		
		# Then we print the results for this epoch:
		print("Epoch "+str(epoch)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
		print("  LR:                            "+str(LR))
		print("  training loss:                 "+str(train_loss))
		print("  validation loss:               "+str(val_loss))
		print("  validation error rate:         "+str(val_err)+"%")
		print("  best epoch:                    "+str(best_epoch))
		print("  best validation error rate:    "+str(best_val_err)+"%")
		print("  test loss:                     "+str(test_loss))
		print("  test error rate:               "+str(test_err)+"%") 
		

		with open("{0}_lr{1}_{2}.txt".format(name,  LR_start, method), "a") as myfile:
			myfile.write("{0}  {1:.5f} {2:.5f} {3:.5f} {4:.5f} {5:.5f} {6:.5f} {7:.5f}\n".format(epoch, 
				train_loss, val_loss, test_loss, val_err, test_err, epoch_duration, LR))


		## Learning rate update scheme
		if name=="mnist":
			if epoch == 15 or epoch==25:
				LR*=LR_decay
		elif name=="cifar":
			if epoch % 15 ==0:
				LR*=LR_decay
		elif name=="svhn":
			if epoch == 15 or epoch==25:
				LR *=LR_decay