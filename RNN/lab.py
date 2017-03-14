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

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
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

### binarize LSTM
class LSTMLayer(lasagne.layers.LSTMLayer):
	
	def __init__(self, incoming, num_units, method, **kwargs):
		
		self.method = method
		g_init = 0.08
		if self.method!="FPN":
			super(LSTMLayer, self).__init__(incoming, num_units, 
				ingate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init)),
					b=lasagne.init.Constant(1.)), 
				cell=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
				outgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				**kwargs)
			# super(LSTMLayer, self).__init__(incoming, num_units, 
			#     **kwargs)
			# add the binary tag to weights            
			self.params[self.W_in_to_ingate]=set(['binary'])
			self.params[self.W_hid_to_ingate]=set(['binary'])

			self.params[self.W_in_to_forgetgate]=set(['binary'])
			self.params[self.W_hid_to_forgetgate]=set(['binary'])

			self.params[self.W_in_to_cell]=set(['binary'])
			self.params[self.W_hid_to_cell]=set(['binary'])

			self.params[self.W_in_to_outgate]=set(['binary'])
			self.params[self.W_hid_to_outgate]=set(['binary'])

		else:
			# super(LSTMLayer, self).__init__(incoming, num_units, **kwargs)
			super(LSTMLayer, self).__init__(incoming, num_units, 
				ingate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init)),
					b=lasagne.init.Constant(1.)), 
				cell=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
				outgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				**kwargs)


		# initialize 2nd moment
		self.acc_W_in_to_ingate = theano.shared(np.zeros((self.W_in_to_ingate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_ingate = theano.shared(np.zeros((self.W_hid_to_ingate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_forgetgate = theano.shared(np.zeros((self.W_in_to_forgetgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_forgetgate = theano.shared(np.zeros((self.W_hid_to_forgetgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_cell = theano.shared(np.zeros((self.W_in_to_cell.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_cell = theano.shared(np.zeros((self.W_hid_to_cell.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_outgate = theano.shared(np.zeros((self.W_in_to_outgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_outgate = theano.shared(np.zeros((self.W_hid_to_outgate.get_value(borrow=True)).shape, dtype='float32'))
		
		self.params[self.acc_W_in_to_ingate]=set(['acc'])
		self.params[self.acc_W_hid_to_ingate]=set(['acc'])

		self.params[self.acc_W_in_to_forgetgate]=set(['acc'])
		self.params[self.acc_W_hid_to_forgetgate]=set(['acc'])

		self.params[self.acc_W_in_to_cell]=set(['acc'])
		self.params[self.acc_W_hid_to_cell]=set(['acc'])

		self.params[self.acc_W_in_to_outgate]=set(['acc'])
		self.params[self.acc_W_hid_to_outgate]=set(['acc'])

	def get_output_for(self, input, deterministic=False, **kwargs):
		# self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
		self.bW_in_to_ingate = binarization(self.W_in_to_ingate, self.acc_W_in_to_ingate, self.method)
		self.bW_hid_to_ingate = binarization(self.W_hid_to_ingate, self.acc_W_hid_to_ingate, self.method)

		self.bW_in_to_forgetgate = binarization(self.W_in_to_forgetgate, self.acc_W_in_to_forgetgate, self.method)
		self.bW_hid_to_forgetgate = binarization(self.W_hid_to_forgetgate, self.acc_W_hid_to_forgetgate, self.method)

		self.bW_in_to_cell = binarization(self.W_in_to_cell, self.acc_W_in_to_cell, self.method)
		self.bW_hid_to_cell = binarization(self.W_hid_to_cell, self.acc_W_hid_to_cell, self.method)

		self.bW_in_to_outgate = binarization(self.W_in_to_outgate, self.acc_W_in_to_outgate, self.method)
		self.bW_hid_to_outgate = binarization(self.W_hid_to_outgate, self.acc_W_hid_to_outgate, self.method)

		# Wr = self.W
		rW_in_to_ingate = self.W_in_to_ingate
		rW_hid_to_ingate = self.W_hid_to_ingate

		rW_in_to_forgetgate = self.W_in_to_forgetgate
		rW_hid_to_forgetgate = self.W_hid_to_forgetgate

		rW_in_to_cell = self.W_in_to_cell
		rW_hid_to_cell = self.W_hid_to_cell

		rW_in_to_outgate = self.W_in_to_outgate
		rW_hid_to_outgate = self.W_hid_to_outgate

		#self.W = bW
		self.W_in_to_ingate = self.bW_in_to_ingate
		self.W_hid_to_ingate = self.bW_hid_to_ingate

		self.W_in_to_forgetgate = self.bW_in_to_forgetgate
		self.W_hid_to_forgetgate = self.bW_hid_to_forgetgate

		self.W_in_to_cell = self.bW_in_to_cell
		self.W_hid_to_cell = self.bW_hid_to_cell

		self.W_in_to_outgate = self.bW_in_to_outgate
		self.W_hid_to_outgate = self.bW_hid_to_outgate
		
		rvalue = super(LSTMLayer, self).get_output_for(input, **kwargs)
		
		self.W_in_to_ingate = rW_in_to_ingate
		self.W_hid_to_ingate = rW_hid_to_ingate

		self.W_in_to_forgetgate = rW_in_to_forgetgate
		self.W_hid_to_forgetgate = rW_hid_to_forgetgate

		self.W_in_to_cell = rW_in_to_cell
		self.W_hid_to_cell = rW_hid_to_cell

		self.W_in_to_outgate = rW_in_to_outgate
		self.W_hid_to_outgate = rW_hid_to_outgate
		
		return rvalue



def compute_grads(loss,network):
		
	layers = lasagne.layers.get_all_layers(network)
	grads = []
	
	for layer in layers:
	
		params = layer.get_params(binary=True)
		if params:
			# print(params[0].name)
			grads.append(theano.grad(loss, wrt=layer.bW_in_to_ingate))
			grads.append(theano.grad(loss, wrt=layer.bW_hid_to_ingate))

			grads.append(theano.grad(loss, wrt=layer.bW_in_to_forgetgate))
			grads.append(theano.grad(loss, wrt=layer.bW_hid_to_forgetgate))

			grads.append(theano.grad(loss, wrt=layer.bW_in_to_cell))
			grads.append(theano.grad(loss, wrt=layer.bW_hid_to_cell))

			grads.append(theano.grad(loss, wrt=layer.bW_in_to_outgate))
			grads.append(theano.grad(loss, wrt=layer.bW_hid_to_outgate))

	return grads


def clipping_scaling(updates,network):
	
	layers = lasagne.layers.get_all_layers(network)
	updates = OrderedDict(updates)
	
	for layer in layers:	
		params = layer.get_params(binary=True)
		for param in params:
			updates[param] = T.clip(updates[param], -1.,1.)     
	return updates
 
# Given a dataset and a model, this function trains the model on the dataset for several epochs
def train(name,method,train_fn,val_fn,
			batch_size,
			SEQ_LENGTH,
			N_HIDDEN,
			LR_start,LR_decay,
			num_epochs,
			X_train,
			X_val,
			X_test):

	def gen_data(pp, batch_size,SEQ_LENGTH, data, return_target=True):

		x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))   ###### 128*100*85
		y = np.zeros((batch_size, SEQ_LENGTH))

		for n in range(batch_size):
			# ptr = n
			for i in range(SEQ_LENGTH):
				x[n,i,char_to_ix[data[pp[n]*SEQ_LENGTH+i]]] = 1.
				y[n,i] = char_to_ix[data[pp[n]*SEQ_LENGTH+i+1]]
		return x, np.array(y,dtype='int32')    

	in_text = X_train+X_val+X_test
	chars = list(set(in_text))
	data_size, vocab_size = len(in_text), len(chars)
	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }
	
	def train_epoch(X,LR):
		
		loss = 0        
		batches = len(X)/batch_size/SEQ_LENGTH
        # shuffle
		num_seq = len(X)/SEQ_LENGTH
		shuffled_ind = range(num_seq)

		np.random.shuffle(shuffled_ind)
		for i in range(batches):
			# shuffle 
			tmp_ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
			xx,yy = gen_data(tmp_ind,batch_size,SEQ_LENGTH, X)
			new_loss, Wg = train_fn(xx,yy,LR)
			loss+=new_loss
		
		loss=loss/batches
		
		return loss
	
	# This function tests the model a full epoch (on the whole dataset)
	def val_epoch(X):
		
		# err = 0
		loss = 0
		batches = len(X)/batch_size/SEQ_LENGTH

		num_seq = len(X)/SEQ_LENGTH
		ind = range(num_seq)
		for i in range(batches):
			tmp_ind = ind[i*batch_size:(i+1)*batch_size]
			xx, yy = gen_data(tmp_ind, batch_size, SEQ_LENGTH, X)
			new_loss = val_fn(xx,yy)
			loss += new_loss
		
		loss = loss/batches

		return loss
	
	best_val_loss=100
	best_epoch = 1
	LR = LR_start
	# hello= False
	# We iterate over epochs:
	for epoch in range(1,num_epochs+1):		
		start_time = time.time()
		train_loss = train_epoch(X_train, LR)
		# try_it_out()

		val_loss = val_epoch(X_val)
		
		# test if validation error went down
		if val_loss <= best_val_loss:
			
			best_val_loss = val_loss
			best_epoch = epoch+1
			
			test_loss = val_epoch(X_test)
		
		epoch_duration = time.time() - start_time
		# Then we print the results for this epoch:
		print("  Epoch "+str(epoch)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
		print("  LR:                            "+str(LR))
		print("  training loss:                 "+str(train_loss))
		print("  validation loss:               "+str(val_loss))
		print("  best epoch:                    "+str(best_epoch))
		print("  test loss:                     "+str(test_loss))
		
		with open("{0}_seq{1}_lr{2}_hid{3}_{4}.txt".format(name, SEQ_LENGTH, LR_start, N_HIDDEN, method), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}\n".format(epoch, train_loss, val_loss, test_loss, epoch_duration))

		# learning rate update scheme
		if epoch>10:
			LR *= LR_decay

