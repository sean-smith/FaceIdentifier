from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


import numpy as np
import cv2
# from matplotlib import pyplot as plt

class Recognize:

	def __init__(self):
		self.net = self.buildNet()
		people = []
		for i in range(1, 6 + 1):
			people.append('Person'+str(i))
		self.classify = ClassificationDataSet(10304, nb_classes=6, class_labels=people)
		print self.classify

	# This call returns a network that has two inputs, three hidden and a single output neuron. 
	# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.	
	def buildNet(self):
		print "Building a network..."
		return buildNetwork(10304, 3, 6)

	def activate(self, img):
		print "Activating a network..."
		i = self.net.activate(img.flatten())
		print i
		return i

	def trainer(self):
		trainer = BackpropTrainer( self.net, dataset=self.classify, momentum=0.1, verbose=True, weightdecay=0.01)
		for i in range(20):
			trainer.trainEpochs( 1 )

	def train_network(self):
		trainer = BackpropTrainer( self.net, dataset=self.classify, momentum=0.1, verbose=True, weightdecay=0.01)
		for i in range(20):
			trainer.trainEpochs( 1 )


	def train(self, img, num):
		print "Training...", "Person = "+str(num)
		self.classify.addSample(img.flatten(), [num])

	def train_images(self, start, stop, num_images):
		# Load a person and dsiplay
		# press n to go to next person
		# press j to go to next image of same person
		for i in range(start, stop+1):
			for num in range(1,num_images + 1):
				img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm', 0)
				height, width = img.shape
				print "height = "+str(height)+" width = "+str(width)


				# Train the neural net
				self.train(img, i)

	def get_max(self, list):
		return reduce(lambda i,x,j,y: i if x > y else j, enumerate(list))

	def neuron(self, edges, input):
		return np.dot(edges.transpose() , input)

	def main(self):

		self.train_images(1, 5, 7)
		self.train_network()

		# edges = np.array([[-1.5, -.5], [1, 1],[1, 1]])
		# input = np.array([[1], [0], [1]])

		# print input 
		# # print edges
		# print edges.transpose()
		

if __name__ == "__main__":
	m = Recognize()
	m.main()
	m.train_network()
