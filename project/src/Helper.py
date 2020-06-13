#torch model helpers


import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
import cv2
from glob import glob
import torch
import pandas as pd
from torch.autograd import Variable
import os
import datetime


print_every = 25
RS = 22 # random seed

def get_cv2_image(path, img_w, img_h, color_type=3):
	# Loading as Grayscale image
	if color_type == 1:
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	elif color_type == 3:
		img = cv2.imread(path, cv2.IMREAD_COLOR)
	# Reduce size
	img = cv2.resize(img, (img_w, img_h)) 
	return img


def load_train(img_w, img_h):
	Xtrain = []
	Ytrain = []
	# Loop over the training folder 
	for x in range(10):
		files = glob(os.path.join('data', 'imgs', 'train', 'c' + str(x), '*.jpg'))
		for file in files:
			img = get_cv2_image(file, img_w, img_h)
			Xtrain.append(img)
			Ytrain.append(x) # class label
	return shuffle(np.asarray(Xtrain), random_state = RS), shuffle(np.asarray(Ytrain), random_state = RS)


def load_test(img_w, img_h):
	Xtest, test_ids = [], []
	# Loop over the training folder 
	files = glob(os.path.join('data', 'imgs', 'test', '*.jpg'))
	for file in files:
		img = get_cv2_image(file, img_w, img_h)
		Xtest.append(img)
		base_file = os.path.basename(file)
		test_ids.append(base_file)

	return shuffle(np.asarray(Xtest), random_state = RS), test_ids


def reset(m):
	if hasattr(m, 'reset_parameters'):
		m.reset_parameters()


def train(model, Xtrain, Ytrain, loss_fn, optimizer, num_epochs, dtype):
	batch_size = 50
	loss_history = []
	for epoch in range(num_epochs):
		print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
		model.train()
		
		batch_amount = int(np.floor(Xtrain.shape[0]/batch_size))
		
		for t in range(batch_amount):
			x = Xtrain[t*batch_size:(t+1)*batch_size]
			y = Ytrain[t*batch_size:(t+1)*batch_size]
			
			x_var = Variable(x.type(dtype))
			y_var = Variable(y.type(dtype).long())

			scores = model(x_var)
			loss = loss_fn(scores, y_var)

			if (t + 1) % print_every == 0:
				print('t = %d, loss = %.4f' % (t + 1, loss.item()))
				loss_history.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	return loss_history

			
def evaluate(model, Xdata, Ydata, length, dtype):  
	num_correct = 0
	num_samples = 0
	model.eval()
	
	batch_size = 50
	batch_amount = int(np.floor(length/batch_size))
	
	for t in range(batch_amount):
		x = Xdata[t*batch_size:(t+1)*batch_size]
		y = Ydata[t*batch_size:(t+1)*batch_size]
		
		with torch.no_grad():
			x_var = Variable(x.type(dtype))

		scores = model(x_var)
		_, preds = scores.data.cpu().max(1)
		num_correct += (preds == y).sum()
		num_samples += preds.size(0)
	
	acc = float(num_correct) / num_samples
	print(f'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
	return 100 * acc

	
def multiclass_log_loss(model, Xdata, Ydata, length, dtype):  
	M = 10 # number of classifications (c0, c1, c2...)
	

	model.eval()
	
	batch_size = 50
	batch_amount = int(np.floor(length/batch_size))
	
	logloss = 0
	
	for t in range(batch_amount):
		x = Xdata[t*batch_size:(t+1)*batch_size]
		y = Ydata[t*batch_size:(t+1)*batch_size]
		
		with torch.no_grad():
			x_var = Variable(x.type(dtype))
	   
		p = nn.Softmax(dim=1)(model(x_var).data.cpu())
				
		for i in range(batch_size):
			for j in range(M):
				y_ij = 1 if y[i] == j else 0
				logloss +=  y_ij * torch.log(p[i][j])

		
	logloss *= -1/length	
	
	print(logloss)
	
	return logloss
	
	
def get_predictions(model, Xdata, length, dtype):
	batch_size = 50
	batch_amount = int(np.ceil(length/batch_size))
	all_preds = []
	model.eval() # put model into test mode
	for t in range(batch_amount): # load in batches to prevent CUDA memory error
		x = Xdata[t*batch_size:(t+1)*batch_size]
		with torch.no_grad():
			x_var = Variable(x.type(dtype))

		scores = model(x_var)
		_, preds = scores.data.cpu().max(1)
		all_preds.extend(np.asarray(preds))

	return all_preds


def create_submission(predictions, test_id):
	# creates a submission for testing accuracy on kaggle
	result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
	result.loc[:, 'img'] = pd.Series(test_id, index=result.index)

	now = datetime.datetime.now()

	if not os.path.isdir('kaggle_submissions'):
		os.mkdir('kaggle_submissions')

	suffix = "{}".format(str(now.strftime("%Y-%m-%d-%H-%M")))
	sub_file = os.path.join('kaggle_submissions', 'submission_' + suffix + '.csv')

	result.to_csv(sub_file, index=False)

	return sub_file


def to_prediction_list(p):
	num_classes = len(np.unique(p))
	pred_matrix = np.zeros((len(p),num_classes))
	pred_matrix[np.arange(len(p)),p] = 1 # set classification index on every row to the predicted value
	return pred_matrix


