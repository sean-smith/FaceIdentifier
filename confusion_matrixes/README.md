# Confusion Matrixes

This folder has two different sub folders

### att_40_classes/  

This is the ATT dataset which has been pre-processed (data is already cropped and grayscale). There are 40 different people each with 10 different images. We split the data 80% meaning that the neural net was trained on 8 images and tested on 2 images for each of the 40 classes. 

Input Images:
	
	1 <= PERSON_NUMBER <= 40
	1 <= IMAGE_NUMBER <= 10
	faces/s[PERSON_NUMBER]/[IMAGE_NUMBER].pgm

### caltech_19_classes/

This is the caltech dataset which came as full color, uncropped images. The data is preprocessed, first it is cropped (using Haar Cascades) then it is scaled to 150x150 pixels and converted to greyscale. Then the image is normalized using Histogram Equalization. The `labeled_faces` is the original labeled data, the `cropped` is the cropped images that are scaled and turned into greyscale. The `cropped_hq` is the cropped and color normalized images.

Input Images:	

	1 <= PERSON_NUMBER <= 19
	0 <= IMAGE_NUMBER <= 9
	labeled_faces/[PERSON_NUMBER]_[IMAGE_NUMBER].jpg
	labeled_faces/cropped/[PERSON_NUMBER]_[IMAGE_NUMBER].jpg
	labeled_faces/cropped_hq/[PERSON_NUMBER]_[IMAGE_NUMBER].jpg

### digits_10_classes/

This is the digits dataset from sk_learn. This is handwritten images of digits (0-9) so 10 classes to classify. Each image is 8x8 pixels There are 1797 images total and approximately 180 images per class.

Input Images:

	sklearn.datasets.load_digits(n_class=10)



