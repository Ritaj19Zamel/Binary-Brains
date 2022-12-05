import numpy
import cv2

image_number=1

while(image_number < 4):
    img = cv2.imread(('C:\\Users\\Zakaria Ahmed\\Downloads\\Graduation Project\\Pothole-Detection\\pothole_')+str(image_number)+'.jpeg')#image1
    # cv2.imshow('image',img)
    blur = cv2.blur(img,(5,5))
    gblur = cv2.GaussianBlur(img,(5,5),0)
    median = cv2.medianBlur(img,5)
    kernel = numpy.ones((5,5),numpy.uint8)
    erosion = cv2.erode(median,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 5)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closing,9,420)
    edges = cv2.Canny(dilation,9,120)

    ret,threshold=cv2.threshold(edges.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img,contours,-1,(0,0,255),2)
    cv2.imshow("Show",img)
    image_number +=1
    cv2.waitKey()#to hold image