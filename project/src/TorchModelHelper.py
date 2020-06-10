#torch model helpers

import numpy as np
from sklearn.utils import shuffle
import cv2
import glob

print_every = 25

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
	Xtest = []
	# Loop over the training folder 
	files = glob(os.path.join('data', 'imgs', 'test', '*.jpg'))
	for file in files:
		img = get_cv2_image(file, img_w, img_h)
		Xtest.append(img)
	return shuffle(np.asarray(Xtest), random_state = RS)


def reset(m):
	if hasattr(m, 'reset_parameters'):
		m.reset_parameters()


def train(model, loss_fn, optimizer, num_epochs = 1):
	for epoch in range(num_epochs):
		print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
		model.train()
		
		batch_size = 50
		batch_amount = int(np.floor(Xtrain.shape[0]/batch_size))
		
		for t in range(batch_amount):
			x = Xtrain[t*batch_size:(t+1)*batch_size]
			y = Ytrain[t*batch_size:(t+1)*batch_size]
			
			x_var = Variable(x.type(DTYPE))
			y_var = Variable(y.type(DTYPE).long())

			scores = model(x_var)
			loss = loss_fn(scores, y_var)
			
			if (t + 1) % print_every == 0:
				print('t = %d, loss = %.4f' % (t + 1, loss.item()))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	return loss_history

			
def evaluate(model, Xdata, Ydata, length):  
	num_correct = 0
	num_samples = 0
	model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
	
	batch_size = 50
	batch_amount = int(np.floor(length/batch_size))
	
	for t in range(batch_amount):
		x = Xdata[t*batch_size:(t+1)*batch_size]
		y = Ydata[t*batch_size:(t+1)*batch_size]
		
		with torch.no_grad():
			x_var = Variable(x.type(DTYPE))

		scores = model(x_var)
		_, preds = scores.data.cpu().max(1)
		num_correct += (preds == y).sum()
		num_samples += preds.size(0)
	
	acc = float(num_correct) / num_samples
	print(f'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
	return 100 * acc


