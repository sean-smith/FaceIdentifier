import cv2
import numpy
import csv
import os
import scipy.sparse.linalg as lin
from numpy.linalg import matrix_rank
from numpy import linalg as LA

train_loc = [] # List of paths of training set
train_class = [] # List of classes (0,1,...,n) of the training set
test_loc = [] # List of paths of testing set
test_class = [] # List of classes (0,1,...,n) of the testing set
train_array = [] # List of N^2x1 vectors corresponding to each face
u = [] # Extended eigenvectors M
u_reduced = [] # Reduced eigenvectors to K = 75% M


def read_csv():
    # This function updates train_loc, train_class, test_loc and test_class
    # img = cv2.imread("/home/davicho/Documents/qt-workspace/CS585/Homeworks/FaceIdentifier/faces/s1/1.pgm")
    # img_flat = img.flatten()

    for person in range(1,41):
        for num in range(1,11):
            path = 'faces/s'+str(person)+'/'+str(num)+'.pgm'

            train_loc.append(path)
            train_class.append(person)

    print train_class
    return train_loc, train_class

    # with open('listFaces.csv', 'rb') as csvfile:
    #     spamreader = csv.reader(csvfile)
    #     num = 1
    #     for row in spamreader:
    #         value = row[0].split(';')
    #         # print str(value[0]) + os.linesep
    #         flag = 1
    #         # if num > 8:
    #         #     flag = 0
    #         if flag == 1:
    #             train_loc.append(value[0])
    #             train_class.append(value[1])
    #         else:
    #             test_loc.append(value[0])
    #             test_class.append(value[1])
    #         num += 1
    #         if num == 11:
    #             num = 1
    # print train_class
    # print test_class
    # return train_loc, train_class, test_loc, test_class

def read_train_images(_data,_class):
    # Read each image
    for i in _data:
        img = cv2.imread(i,0)
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
    # This line will help us to define how many eigenvectors we
    # can calculate. # of eigs = rank -1 
    print matrix_rank(cov)
    eigval, eigvec = lin.eigs(cov, 38)
    # eigval, eigvec = LA.eig(cov)
    print len(eigval)
    print "size of the eigen vector " + str(len(eigvec[:, 0]))
    print "size of the eigen vector matrix " + str(len(eigvec))
    print "number of eigenvalues " + str(len(eigval))
    # Each eigvec[:,i] is an eigenvector of size 40
    # We need to find u_i = A * eigvec[:,i]
    A = norm_train_array.T
    for i in range(1,(len(eigvec[0]+1))):
        u.append(numpy.dot(A,eigvec[:,i]))
    for i, val in enumerate(u):
        u[i] = u[i] / numpy.linalg.norm(u[i])
    # We're only keeping 75% of the number of eigenvector u[i]
    # This will correspond to the largest eigenvalues
    # First we will sort our eigenvalues 
    real_eigval = []
    for i in range(len(eigval)):
        print i
        real_eigval.append(eigval[i].real)
    real_eigval = numpy.asarray(real_eigval)
    idx = real_eigval.argsort()[::-1]   
    sorted_eigval = real_eigval[idx]
    sorted_u = numpy.empty(len(u))
    print u[idx]
    # sorted_u = u[:,idx]
    # sorted_eigval = sorted(eigval.real)
    # sorted_index = sorted(range(len(eigval.real)), key=lambda k: eigval[k])
    # print sorted_eigval
    # print sorted_index

    for i in range(1,(int(0.75*len(u))+4)):
        u_reduced.append(u[i])
    # u_reduced[i] are called Eigenfaces
    # Now lets represent each face in this basis
    sigma = []
    omega = []
    for i, val in enumerate(list_norm_train):
        sigma = []
        for j, val in enumerate(u_reduced):
            w = numpy.dot(u_reduced[j].T,list_norm_train[i])
            sigma.append(w.real)
            sigma_array = numpy.asarray(sigma)
        omega.append(sigma_array)
    #print omega
    print "size of eigenvector" + str(len(omega[0]))
    print "size of omega" + str(len(omega))
    #return eigval, eigvec
    return omega, train_array, u, u_reduced

# def visualize_eigenfaces(_eigenfaceMatrix):
#     # Here we input our eigenfaces that will classify our data
#     for i, val in enumerate(_eigenfaceMatrix):


if __name__ == "__main__":
    read_csv()
    #val, vec = read_train_images(train_loc, train_class)
    omega = read_train_images(train_loc, train_class)
    print 'Finished'
    #print train_array
    #print mean/len(mean)
    #print val
    #print vec
    

