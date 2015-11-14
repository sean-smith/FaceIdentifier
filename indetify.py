from pybrain.tools.shortcuts import buildNetwork

# This call returns a network that has two inputs, three hidden and a single output neuron. 
# In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.
net = buildNetwork(2, 3, 1)


net.activate([2, 1])

