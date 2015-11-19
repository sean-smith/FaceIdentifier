import cv2
import numpy
import csv
import os
import scipy.sparse.linalg as lin

train_loc = []
train_class = []
test_loc = []
test_class = []
train_array = []
#mean = []


def read_csv():
    # img = cv2.imread("/home/davicho/Documents/qt-workspace/CS585/Homeworks/FaceIdentifier/faces/s1/1.pgm")
    # img_flat = img.flatten()
    with open('listFaces.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        num = 1
        for row in spamreader:
            value = row[0].split(';')
            # print str(value[0]) + os.linesep
            flag = 1
            if num > 8:
                flag = 0
            if flag == 1:
                train_loc.append(value[0])
                train_class.append(value[1])
            else:
                test_loc.append(value[0])
                test_class.append(value[1])
            num += 1
            if num == 11:
                num = 1
    # print train_class
    # print test_class

def read_train_images(_data,_class):
    # Read each image
    for i in _data:
        img = cv2.imread(i,0)
        img_flat = img.flatten()
        train_array.append(img_flat)
    # Calculate the mean of the set
    mean = [sum(i) for i in zip(*train_array)]
    length = len(_data)
    mean = [float(i) / length for i in mean]
    #mean = [i / length for i in mean]
    # Substract mean to each image (this can be negative so python 
    # transform each vector to int)
    list_norm_train = [i - mean for i in train_array]
    # Transform the list to an array
    norm_train_array = numpy.asarray(list_norm_train)
    # Compute the covariance matrix of the array
    cov = numpy.dot(norm_train_array,norm_train_array.T)
    eigval, eigvec = lin.eigs(cov, 38)
    #return cov
    return eigval, eigvec

if __name__ == "__main__":
    read_csv()
    val, vec = read_train_images(train_loc,train_class)
    print 'Finished'
    #print train_array
    #print mean/len(mean)
    print val
    print vec
    

