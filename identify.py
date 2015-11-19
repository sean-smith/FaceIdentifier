from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


import numpy as np
import cv2
# from matplotlib import pyplot as plt

class Recognize:

	def __init__(self):
		self.net = self.buildNet()
		self.classify = self.classify()
		self.trainer = self.trainer()
		

	# This call returns a network that has 10304 inputs, three hidden and 40 output neurons. 
	# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.	
	def buildNet(self):
		print "Building a network..."
		return buildNetwork(10304, 3, 40)

	def classify(self):
		self.classify = ClassificationDataSet(10304, target=40, nb_classes=40)

		self.train_images()

		print "Number of training patterns: ", len(self.classify)
		print "Input and output dimensions: ", self.classify.indim, self.classify.outdim
		return self.classify


	def identify(self, i):
		for num in range(1,11):
			img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
			l = self.net.activate(img.flatten())
			i = [i for i, x in enumerate(l) if x > l[i - 1]]

			print "i = "+str(i)+" = "+str()


	def trainer(self):
		print "self.net.outdim = "+str(self.net.outdim)
		print "self.classify.outdim = "+str(self.classify.outdim)

		trainer = BackpropTrainer(self.net, dataset=self.classify, momentum=0.1, verbose=True, weightdecay=0.01)
		# trainer.trainUntilConvergence()

		for i in range(5):
			trainer.trainEpochs( 1 )
			trnresult = percentError( trainer.testOnClassData(), self.classify['class'])
			tstresult = percentError( trainer.testOnClassData( dataset=self.classify ), self.classify['class'] )

			print "epoch: %4d" % trainer.totalepochs, \
				"  train error: %5.2f%%" % trnresult, \
				"  test error: %5.2f%%" % tstresult
		return trainer

	def train_images(self):
		for i in range(1, 41):
			# Create a Binary List
			l = [0] * 40
			for num in range(1,11):
				img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
				height, width = img.shape
				print "Training...", "Person = "+str(i)
				l[i - 1] = 10000
				self.classify.addSample(img.flatten(), l)
				#self.classify.addSample(img.flatten(), [i])

	def neuron(self, edges, input):
		return np.dot(edges.transpose(), input)
		

if __name__ == "__main__":
	m = Recognize()
	m.identify(1)
	m.identify(2)
	m.identify(3)

