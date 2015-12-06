# import cv2

# # rectangle color and lineWidth
# color = (0, 0, 255)
# lineWidth = 1

# windowName = "Object Detection"

# # load detection file (various files for different views and uses)
# cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# indexes = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12]

# for i in range(0, 10):
#     # load an image to search for faces
#     idx = '0' + str(indexes[i])
#     if indexes[i] < 10:
#         idx = '00' + idx
#     elif (i >= 10 and i < 100):
#         idx = '0' + idx
#     face = "./faces2/image_" + idx + ".jpg"
#     print face
#     img = cv2.imread(face)

#     # detect objects, return as list
#     rects = cascade.detectMultiScale(img)
#     print rects

#     for x,y, width,height in rects:
#         cv2.rectangle(img, (x,y), (x+width, y+height), color, lineWidth)
#         roi  = img[y:(y+height), x:(x+width)]
#         gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         gray_image = cv2.resize(gray_image, (150, 150))
#         # equ = cv2.equalizeHist(gray_image)
#     # cropped = './cropped/image_' + idx + '.jpg'
#     # cv2.imwrite(cropped,gray_image)
#     # cropped2 = './cropped2/image_' + idx + '.jpg'
#     # cv2.imwrite(cropped2,equ)

import os
import shutil
def copy_rename(old_file_name, new_file_name):
        src_dir= os.path.join(os.curdir , "unlabeled_faces")
        print "src_dir", src_dir
        dst_dir= os.path.join(os.curdir , "labeled_faces")
        print "dst_dir", dst_dir
        src_file = os.path.join(src_dir, old_file_name)
        print "src_file", src_file
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        print "dst_file", dst_file
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        print new_dst_file_name
        os.rename(dst_file, new_dst_file_name)


def main():
    indexes = [i for i in range(441, 450+1)]

    for i in range(0, 10):
        # load an image to search for faces
        idx = '0' + str(indexes[i])
        if indexes[i] < 10:
            idx = '00' + idx
        elif (i >= 10 and i < 100):
            idx = '0' + idx
        face = "image_" + idx + ".jpg"
        new_face = '12_0' + str(i) + ".jpg"
        copy_rename(face,new_face)

if __name__ == "__main__":
    main()