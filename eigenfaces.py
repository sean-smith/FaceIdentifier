import cv2
import numpy
import csv
import os
import scipy.sparse.linalg as lin
from numpy.linalg import matrix_rank
from numpy import linalg as LA
import scipy.io
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pylab

train_loc = []  # List of paths of training set
train_class = []  # List of classes (0,1,...,n) of the training set
test_loc = []  # List of paths of testing set
test_class = []  # List of classes (0,1,...,n) of the testing set
train_array = []  # List of N^2x1 vectors corresponding to each face
u = []  # Extended eigenvectors M
u_reduced = []  # Reduced eigenvectors to K = 75% M


def read_csv():
    # This function updates train_loc, train_class, test_loc and test_class
    # img = cv2.imread("/home/davicho/Documents/qt-workspace/CS585/Homeworks/FaceIdentifier/faces/s1/1.pgm")
    # img_flat = img.flatten()

    for person in range(1,41):
        for num in range(1, 11):
            path = 'faces/s'+str(person)+'/'+str(num)+'.pgm'

            train_loc.append(path)
            train_class.append(person)

    print train_class
    return train_loc, train_class


def read_train_images(_data, _class):
    # Read each image
    for i in _data:
        img = cv2.imread(i, 0)
        img_flat = img.flatten()
        # Build our train_array list
        train_array.append(img_flat)
    # Calculate the mean of the set
    mean = [sum(i) for i in zip(*train_array)]
    length = len(_data)
    mean = [float(i) / length for i in mean]
    # Substract mean to each image (this can be negative so python
    # transform each vector to int)
    # This is Omega
    list_norm_train = [i - mean for i in train_array]
    # Transform the list to an array
    norm_train_array = numpy.asarray(list_norm_train)
    # Compute the covariance matrix of the array
    cov = numpy.dot(norm_train_array, norm_train_array.T)
    [mm, nn] = cov.shape
    print "Size of the Cov Matrix is: " + str(mm*nn)
    # print "Rank of the Cov Matrix is: " + str(matrix_rank(cov))
    # print "Number of non zero elements in CM: " + str(numpy.count_nonzero(cov))
    # We're choosing numpy eig function rather than scipy
    # It calculates same # of eigval, eigvec as size of matrix
    # eigval, eigvec = lin.eigs(cov, 48)
    eigval, eigvec = LA.eig(cov)
    eigval = eigval.real
    eigvec = eigvec.real
    # numpy.savetxt('eigval.out', eigval.real, delimiter=',')
    # numpy.savetxt('eigvec.out', eigvec.real, delimiter=',')
    print "Number of Eigenvalues: " + str(len(eigval))
    print "Size of the Eigenvector: " + str(len(eigvec[:, 0]))
    print "Number of Eigenvectors: " + str(len(eigvec))
    # Each eigvec[:,i] is an eigenvector of size 40
    # We need to find u_i = A * eigvec[:,i]
    A = norm_train_array.T
    for i in range(0, len(eigvec[:, 0])):
        u.append(numpy.dot(A, eigvec[:, i]))
    for i, val in enumerate(u):
        u[i] = u[i] / numpy.linalg.norm(u[i])
    print "Size of the new norm eigenvector: " + str(len(u[0]))
    # print "Number of eigenvectors: " + str(len(u))
    # We're only keeping 75% of the number of eigenvector u[i]
    # This will correspond to the largest eigenvalues
    # First we will sort our eigenvalues
    idx = eigval.argsort()[::-1]
    sorted_eigval = eigval[idx]
    # Now we will sort our eigenvectors with the index from our eigenvalues
    sorted_u = []
    for i in range(0, len(idx)):
        sorted_u.append(u[idx[i]])
    # numpy.savetxt('sorted_eigval.out', sorted_eigval, delimiter=',')
    # Now we will save a fraction of the number of eigenvectors
    for i in range(0, int(0.75*len(u))):
        u_reduced.append(sorted_u[i])
    # print "Size of the Reduced Eigenvector: " + str(len(u_reduced[0]))
    print "Number of reduced eigenvectors: " + str(len(u_reduced))
    # u_reduced are called Eigenfaces
    # Now lets represent each face in this basis
    sigma = []
    omega = []
    for i, val in enumerate(list_norm_train):
        sigma = []
        for j, val in enumerate(u_reduced):
            w = numpy.dot(u_reduced[j].T, list_norm_train[i])
            sigma.append(w.real)
            sigma_array = numpy.asarray(sigma)
        omega.append(sigma_array)
    # print omega
    print "Size of basis vector: " + str(len(omega[0]))
    print "Size of Omega: " + str(len(omega))
    # return eigval, eigvec
    return omega, train_array, u, u_reduced


def visualize_eigenfaces(_eigenfaceMatrix, height, width):
    # Here we input our eigenfaces that will classify our data
    # fig = plt.figure()
    fig = pylab.figure()
    length = len(_eigenfaceMatrix)
    for i in range(1, 7):
        r = random.randrange(0, length)
        fig.add_subplot(2, 3, i)
        if i == 2:
            pylab.title('Eigenfaces samples')
        img = numpy.reshape(_eigenfaceMatrix[r], (height, width))
        img = img - numpy.min(img)
        img = img * (1/numpy.max(img))
        pylab.imshow(img, cmap=cm.Greys_r)
    pylab.show()


def extractFace(_img):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(_img)
    for x, y, width, height in rects:
        roi = _img[y:(y+height), x:(x+width)]
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (150, 150))
    return gray_image

if __name__ == "__main__":
    read_csv()
    [omega, t_array, v, v_reduced] = read_train_images(train_loc, train_class)
    height = 112
    width = 92
    visualize_eigenfaces(v_reduced, height, width)
    # aa = numpy.reshape(v_reduced[1], (112, 92))
    # # scipy.io.savemat('arrdata.mat', mdict={'aa': aa})
    # # aa = aa * 256
    # aa = aa - numpy.min(aa)
    # aa = aa * (1/numpy.max(aa))

    # # print numpy.min(aa), numpy.max(aa)
    # cv2.imshow("eigenface", aa)
    # cv2.waitKey(0)

    print 'Finished'
