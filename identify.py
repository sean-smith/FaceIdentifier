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
		

	# This call returns a network that has two inputs, three hidden and a single output neuron. 
	# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.	
	def buildNet(self):
		print "Building a network..."
		return buildNetwork(10304, 3, 40)

	def classify(self):
		people = []
		for i in range(1, 41):
			people.append('Person'+str(i))
		self.classify = ClassificationDataSet(10304, nb_classes=40, class_labels=people)

		tstdata, trndata = self.classify.splitWithProportion( 0.30 )
		for n in xrange(0, tstdata.getLength()):
			tstdata.addSample( tstdata.getSample(n)[0], tstdata.getSample(n)[1] )
		for n in xrange(0, trndata.getLength()):
			trndata.addSample( trndata.getSample(n)[0], trndata.getSample(n)[1] )

		print "Number of training patterns: ", len(trndata)
		print "Input and output dimensions: ", trndata.indim, trndata.outdim
		print "First sample (input, target, class):"
		print trndata['input'][0], trndata['target'][0]
		return trndata


	def activate(self, img):
		print "Activating a network..."
		i = self.net.activate(img.flatten())
		print i
		return i

	def trainer(self):
		print "self.net.outdim = "+str(self.net.outdim)
		print "self.classify.outdim = "+str(self.classify.outdim)

		

		trainer = BackpropTrainer( self.net, dataset=self.classify, momentum=0.1, verbose=True, weightdecay=0.01)
		#t.
		# for i in range(20):
		# 	trainer.trainEpochs( 1 )
		# 	trnresult = percentError( trainer.testOnClassData(), trndata['class'])
		# 	tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )

		# 	print "epoch: %4d" % trainer.totalepochs, \
		# 		"  train error: %5.2f%%" % trnresult, \
		# 		"  test error: %5.2f%%" % tstresult
		

	def train_images(self, start, stop, num_images):
		# Load a person and dsiplay
		# press n to go to next person
		# press j to go to next image of same person
		for i in range(start, stop+1):
			for num in range(1,num_images + 1):
				img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
				height, width = img.shape
				# print "height = "+str(height)+" width = "+str(width)
				print "Training...", "Person = "+str(num)
				self.classify.addSample(img.flatten(), [i])


				# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
				# cv2.imshow('image',img)
				# k = cv2.waitKey(0) & 0xFF
				# if k == ord('j'):
				# 	cv2.destroyAllWindows()
				# elif k == ord('n'):
				# 	i+=1
				# elif k == ord('u'):
				# 	return

	def get_max(self, list):
		return reduce(lambda i,x,j,y: i if x > y else j, enumerate(list))

	def neuron(self, edges, input):
		return np.dot(edges.transpose(), input)

	def main(self):

		result = self.train_images(1, 40, 7)
		print result
		self.trainer()

		# edges = np.array([[-1.5, -.5], [1, 1],[1, 1]])
		# input = np.array([[1], [0], [1]])

		# print input 
		# # print edges
		# print edges.transpose()
		

if __name__ == "__main__":
	m = Recognize()
	m.main()
