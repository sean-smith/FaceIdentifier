import cv2

# rectangle color and lineWidth
color = (0, 0, 255)
lineWidth = 1

windowName = "Object Detection"

# load detection file (various files for different views and uses)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

for i in range(1, 451):
    # load an image to search for faces
    idx = '0' + str(i)
    if i < 10:
        idx = '00' + idx
    elif (i >= 10 and i < 100):
        idx = '0' + idx
    face = "./faces/image_" + idx + ".jpg"
    print face
    img = cv2.imread(face)

    # detect objects, return as list
    rects = cascade.detectMultiScale(img)
    print rects

    for x,y, width,height in rects:
        cv2.rectangle(img, (x,y), (x+width, y+height), color, lineWidth)
        roi  = img[y:(y+height), x:(x+width)]
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (150, 150)) 

    # cv2.imshow(windowName, img)
    cv2.imshow(windowName,gray_image)
    cv2.moveWindow(windowName, 10, 10)

    cv2.waitKey(20)
    cv2.destroyAllWindows()
