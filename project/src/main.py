#main.py
import numpy as np
import os

def avg_number_images_per_set(rootdir : str):
	# returns the average number of images in each data set
	all_sets = os.listdir(rootdir)
	num_set = len(all_sets)
	num_images = 0
	for dataset in all_sets:
		num_images += len(os.listdir(rootdir + dataset))
	return num_images / num_set



