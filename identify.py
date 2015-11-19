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

class Recognize:

	def __init__(self):
		self.trained = False
		print "Enter 1 for data from faces and 2 for sklearn data:"
		self.org = True if int(raw_input()) == 1 else False
		if self.org:
			print "Using Original data from faces/"
		else:
			print "Using data from sklearn"
		self.path = None
		self.x = None
		self.classify = self.classify()
		self.net = self.buildNet()
		self.trainer = self.train()

	# This call returns a network that has 10304 inputs, 64 hidden and 1 output neurons. 
	# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.	
	def buildNet(self):
		print "Building a network..."
		self.path = 'net.xml' if self.org else 'net_sklearn.xml'
		if  os.path.isfile(self.path): 
			self.trained = True
 			return NetworkReader.readFrom(self.path) 
		else:
			dim = 106 if self.org else 64
 			return buildNetwork(self.classify.indim, dim, self.classify.outdim, outclass=SoftmaxLayer)
		

	def classify(self):
		if self.org:
			self.classify = ClassificationDataSet(10304, target=1, nb_classes=40)
			self.train_images()
		else:
			self.classify = ClassificationDataSet(4096, target=1, nb_classes=40)
			self.train_images2()
		
		# convert binary value to a number
		print "Input and output dimensions: ", self.classify.indim, self.classify.outdim

		self.classify._convertToOneOfMany()

		print "Number of training patterns: ", len(self.classify)
		print "Input and output dimensions: ", self.classify.indim, self.classify.outdim
		return self.classify

	def identify(self, i):
		if not self.org:
			self.identify2(i)
			return
		print "Identifying Image 1"
		for num in range(1,11):
			img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
			l = self.net.activate(np.ravel(img))
			max_index, max_value = max(enumerate(l), key=lambda x: x[1])
			print str(i)+"   "+str(max_index), i == max_index
			#print l

	def identify2(self, i):
		for m in range(1,11):
			l = self.net.activate(np.ravel(self.x.data[i * 10 + m]))
			max_index, max_value = max(enumerate(l), key=lambda x: x[1])
			print str(i)+"   "+str(max_index), i == max_index

	def train(self):
		print "Enter the number of times to train, -1 means train until convergence:"
		t = int(raw_input())
		print "Training the Neural Net"
		print "self.net.outdim = "+str(self.net.outdim)
		print "self.classify.outdim = "+str(self.classify.outdim)

		trainer = BackpropTrainer(self.net, dataset=self.classify, momentum=0.1, verbose=True, weightdecay=0.01)
		
		if t == -1:
			trainer.trainUntilConvergence()
		else:
			for i in range(t):
				trainer.trainEpochs(1)
				trnresult = percentError( trainer.testOnClassData(), self.classify['class'])
				tstresult = percentError( trainer.testOnClassData(dataset=self.classify), self.classify['class'] )

				print "epoch: %4d" % trainer.totalepochs, \
					"  train error: %5.2f%%" % trnresult, \
					"  test error: %5.2f%%" % tstresult

		print "Done training... Writing to a file"
		NetworkWriter.writeToFile(self.net, self.path)
		return trainer

	def train_images(self):
		for i in range(1, 41):
			for num in range(1,11):
				img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
				height, width = img.shape
				print "Training...", "Person = "+str(i)
				# print type(np.ravel(img)[0]), type(np.int64(i-1))
				self.classify.addSample(np.ravel(img), np.int64(i-1))

	def train_images2(self):
		self.x = datasets.fetch_olivetti_faces()
		for i in range(41):
			#print type(self.x.data[i]), type(self.x.target[i])
			self.classify.addSample(self.x.data[i], self.x.target[i])
		

if __name__ == "__main__":
	m = Recognize()
	m.identify(1)
	m.identify(2)
	m.identify(3)
	m.identify(4)
	m.identify(5)
	m.identify(6)

