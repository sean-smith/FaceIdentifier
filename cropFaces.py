import cv2

# rectangle color and lineWidth
color = (0, 0, 255)
lineWidth = 1

windowName = "Object Detection"

# load detection file (various files for different views and uses)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

for person in range(1,11):
    for num in range(0, 10):
        facepath = format(person,'02')+'_'+format(num,'02')+'.jpg'
        path = 'faces1/'+ facepath
        img = cv2.imread(path)
        # detect objects, return as list
        rects = cascade.detectMultiScale(img)
        # print rects

        for x,y, width,height in rects:
            cv2.rectangle(img, (x,y), (x+width, y+height), color, lineWidth)
            roi  = img[y:(y+height), x:(x+width)]
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (150, 150))
            equ = cv2.equalizeHist(gray_image)
        # cropped = 'faces1/cropped/' + facepath
        cropped = 'faces1/cropped_hq/' + facepath
        print cropped
        # cv2.imwrite(cropped,gray_image)
        cv2.imwrite(cropped,equ)

