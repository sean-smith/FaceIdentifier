from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import cv2
# from matplotlib import pyplot as plt

def main():
	# Load a person and dsiplay
	# press n to go to next person
	# press j to go to next image of same person
	for i in range(1, 41):
		for num in range(1,11):
			img = cv2.imread('faces/s'+str(i)+'/'+str(num)+'.pgm',1)

			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			cv2.imshow('image',img)
			k = cv2.waitKey(0) & 0xFF
			if k == ord('j'):
				cv2.destroyAllWindows()
			elif k == ord('n'):
				i+=1


# This call returns a network that has two inputs, three hidden and a single output neuron. 
# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.
print "Building a network..."
net = buildNetwork(2, 3, 1)

print "Activating a network..."
print net.activate([2, 1])

edges = np.array([[-1.5, -.5], [1, 1],[1, 1]])

# print edges

print edges.transpose()

input = np.array([[1], [0], [1]])

print input 

def neuron(edges, input):
	return np.dot(edges.transpose() , input)

print neuron(edges, input)

if __name__ == "__main__":
	main()
