from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

import os

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


from sklearn import datasets
import numpy as np
import cv2

import eigenfaces
# import config


class Recognize:
	def __init__(self):
		self.d = {
			"train_numbers.xml": {
				'hidden_dim': 32,
				'nb_classes': 10,
				'in_dim': 64,
				'train_func': self.train_digits,
				'identify_func': self.identify_digits,
			},
			"net_test.xml": {
				'hidden_dim': 150,
				'nb_classes': 40,
				'in_dim': 300,
				'train_func': self.train_images,
				'identify_func': self.identify3,
				'split_percent': .80,
				'faces_per_person': 10,
			},
			"net_color_caltech.xml": {
				'hidden_dim': 150,
				'nb_classes': 19,
				'in_dim': 142,
				'train_func': self.train_images,
				'identify_func': self.identify3,
				'split_percent': .80,
				'faces_per_person': 10,
			},
			"net_color_caltech_eq.xml": {
				'hidden_dim': 150,
				'nb_classes': 19,
				'in_dim': 142,
				'train_func': self.train_images,
				'identify_func': self.identify3,
				'split_percent': .80,
				'faces_per_person': 10,
			},
			"net.xml": {
				'hidden_dim': 106,
				'nb_classes': 40,
				'in_dim': 10304,
				'train_func': self.train_images,
				'identify_func': self.identify1,
			},
			"net_sklearn.xml": {
				'hidden_dim': 64,
				'nb_classes': 40,
				'in_dim': 4096,
				'train_func': self.train_images2,
				'identify_func': self.identify2,
			},
		}
		self.trained = False
		# When changing from net_color_caltech to net_color_caltech_eq change line 42 in eigenfaces.py
		self.path = "net_color_caltech.xml"
		self.x = None
		self.all_data = self.classify()
		self.net = self.buildNet()
		self.trainer = self.train()

	# This call returns a network that has 10304 inputs, 64 hidden and 1 output neurons. 
	# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.	
	def buildNet(self):
		print "Building a network..."
		if  os.path.isfile(self.path): 
			self.trained = True
 			return NetworkReader.readFrom(self.path) 
		else:
 			return buildNetwork(self.all_data.indim, self.d[self.path]['hidden_dim'], self.all_data.outdim, outclass=SoftmaxLayer)
	
	def get_max_index(self, l):
		max_index, max_value = max(enumerate(l), key=lambda x: x[1])
		return max_index

	def classify(self):
		print "self.d[self.path]['in_dim'] = ", self.d[self.path]['in_dim']
		self.all_data = ClassificationDataSet(self.d[self.path]['in_dim'], target=1, nb_classes=self.d[self.path]['nb_classes'])
		self.train_data = ClassificationDataSet(self.d[self.path]['in_dim'], target=1, nb_classes=self.d[self.path]['nb_classes'])
		self.test_data = ClassificationDataSet(self.d[self.path]['in_dim'], target=1, nb_classes=self.d[self.path]['nb_classes'])

		# add data to self.all_data dataset
		self.d[self.path]['train_func']()

		# turns 1 => [0,1,...]
		self.all_data._convertToOneOfMany()
		self.train_data._convertToOneOfMany()
		self.test_data._convertToOneOfMany()


		print "self.all_data.outdim = ", self.all_data.outdim	
		print "Input and output dimensions: ", self.all_data.indim, self.all_data.outdim
		print "Number of training patterns: ", len(self.all_data)
		print "Input and output dimensions: ", self.all_data.indim, self.all_data.outdim
		return self.all_data

	def identify_digits(self, i):
		for image, label in self.images_and_labels:
			if label == i:
				l = self.net.activate(np.ravel(image))
				max_index, max_value = max(enumerate(l), key=lambda x: x[1])
				print str(i)+"   "+str(max_index), i == max_index

	def identify(self, i):
		self.d[self.path]['identify_func'](i)

	# For original Att data
	def identify1(self, i):
		print "Identifying Image 1"
		for num in range(1,11):
			img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
			l = self.net.activate(np.ravel(img))
			max_index, max_value = max(enumerate(l), key=lambda x: x[1])
			print str(i)+"   "+str(max_index), i == max_index
			print l

	# Handwritten Digits
	def identify2(self, i):
		for m in range(1,11):
			l = self.net.activate(np.ravel(self.x.data[i * 10 + m]))
			max_index, max_value = max(enumerate(l), key=lambda x: x[1])
			print str(i)+"   "+str(max_index), i == max_index

	# For caltech images
	def identify3(self, m):
		for i in range(len(self.test_data['input'])):
			img = self.test_data['input'][i]
			label = self.test_data['target'][i]
			l = self.net.activate(img)
			result = self.get_max_index(l)
			label = self.get_max_index(label)
			print str(label)+"\t"+str(result)+"\t"+ str(int(label) == int(result))



	def train(self):
		print "Enter the number of times to train, -1 means train until convergence:"
		t = int(raw_input())
		print "Training the Neural Net"
		print "self.net.indim = "+str(self.net.indim)
		print "self.train_data.indim = "+str(self.train_data.indim)

		trainer = BackpropTrainer(self.net, dataset=self.train_data, momentum=0.1, verbose=True, weightdecay=0.01)
		
		if t == -1:
			trainer.trainUntilConvergence()
		else:
			for i in range(t):
				trainer.trainEpochs(1)
				trnresult = percentError( trainer.testOnClassData(), self.train_data['class'])
				# print self.test_data

				tstresult = percentError( trainer.testOnClassData(dataset=self.test_data), self.test_data['class'] )

				print "epoch: %4d" % trainer.totalepochs, \
					"  train error: %5.2f%%" % trnresult, \
					"  test error: %5.2f%%" % tstresult

				if i % 10 == 0 and i > 1:
					print "Saving Progress... Writing to a file"
					NetworkWriter.writeToFile(self.net, self.path)

		print "Done training... Writing to a file"
		NetworkWriter.writeToFile(self.net, self.path)
		return trainer

	def train_digits(self):
		x = datasets.load_digits()
		self.images_and_labels = list(zip(x.images, x.target))
		for image, label in self.images_and_labels:
			self.all_data.addSample(np.ravel(image), label)


	def train_images(self):
		# Call eigenfaces here
		train_loc, self.train_class = eigenfaces.read_csv()
		self.omega, train_array, u, u_reduced = eigenfaces.read_train_images(train_loc, self.train_class)

		# 0 < split_percent < 1
		# 0 < test_number < 10
		test_number = int(self.d[self.path]['split_percent'] * self.d[self.path]['faces_per_person'])
		print "test_number", test_number

		for i in range(len(self.omega)):
			img = self.omega[i]
			label = self.train_class[i]

			# add data
			self.all_data.addSample(img, int(label)-1)

			if (i % 10) >= test_number:
				self.test_data.addSample(img, int(label) - 1)
			else:
				self.train_data.addSample(img, int(label) - 1)

		print "size of test data", len(self.test_data['input'])
		print "size of train data", len(self.train_data['input'])



	def train_images2(self):
		self.x = datasets.fetch_olivetti_faces()
		for i in range(41):
			self.all_data.addSample(self.x.data[i], self.x.target[i])
		

if __name__ == "__main__":
	m = Recognize()
	m.identify(1)

