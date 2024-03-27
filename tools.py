import cv2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from PIL import Image, ExifTags

from datetime import datetime
from tqdm import tqdm
import re
import pandas as pd


"""
Daniel Cruz
September, 2023

Methods created for horizone rectification, saving images and csv files, detecting bluriness, etc.
"""

#inspired by method developed by Sean Sall origin: https://github.com/sallamander/horizon-detection/blob/master/utils.py
def detect_horizon_line(image_grayscaled):
    """Detect the horizon's starting and ending points in the given image

    The horizon line is detected by applying Otsu's threshold method to
    separate the sky from the remainder of the image.

    :param image_grayscaled: grayscaled image to detect the horizon on, of
     shape (height, width)
    :type image_grayscale: np.ndarray of dtype uint8
     points of the detected horizon line
    """

    msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
           'of shape (height, width).')
    assert image_grayscaled.ndim == 2, msg
    image_blurred = cv2.GaussianBlur(image_grayscaled, ksize=(3, 3), sigmaX=0)

    _, image_thresholded = cv2.threshold(
        image_blurred, thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    image_thresholded = image_thresholded - 1
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                    kernel=np.ones((9, 9), np.uint8))

    horizon_x1 = 0
    horizon_x2 = image_grayscaled.shape[1] - 1
    horizon_y1 = max(np.where(image_closed[:, horizon_x1] == 0)[0])
    horizon_y2 = max(np.where(image_closed[:, horizon_x2] == 0)[0])

    #get x coordinates of the threshold in 10 pixel steps
    x = np.arange(horizon_x1,horizon_x2,10)
    y = []
    #Find y coordinates that land on the edge of the threshold
    for i in x:
        y.append(min(np.where(image_closed[:, i] == 255)[0]))
    return x,y,image_closed

#warps image given a matrix
def warp(img,matrix):
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

#finds rotation matrix based off of angle in degrees
def getRotationMatrix(img,angle):
    width,height = img.shape[:2]
    center = (width//2, height//2)
    return cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

#finds matrix for vertical shifting to align to golden standard. Uses the line of best fit from the golden standard
def getVerticalShiftMatrix(img,a,b,goldA,goldB):
    width,height = img.shape[:2]
    center = (width//2, height//2)
    
    horizonHeight = a*center[1] + b
    goldHorizonHeight = goldA*center[1] + goldB
    
    diff = goldHorizonHeight - horizonHeight
    return np.array([[1,0,0],
                     [0,1,diff]])

    
#retrieves data time from exif
def getDateTimeFromExif(imdir):
    pimg = Image.open(str(imdir))
    img_exif = pimg.getexif()
    #this is the code for datetime
    strdatetime = img_exif[306]
    timestamp = datetime.strptime(strdatetime,"%Y:%m:%d %H:%M:%S")
    return timestamp

#retrieves date time for phenocam names
def getDateTimeFromName(name):
    parts = name.split("_")
    strdatetime = "_".join(parts[1:3])

    #check for timezone
    if "-" in strdatetime or "+" in strdatetime:
        timestamp = datetime.strptime(strdatetime,"%Y%m%d_%H%M%S%z")
    else:
        timestamp = datetime.strptime(strdatetime,"%Y%m%d_%H%M%S")
    return timestamp


#returns directory structure with parent directories removed
def getfilestructure(path):
    return str(path.parent).replace(str(data_dir),"")[1:] #indexing to remove starting /

#writes a single row to a csv file, row and title are expected to be a list
def writetocsv(filename,row,title):
    import csv
    if not filename.is_file(): #write title if file is new
        with open(filename, '+a') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(title)
            f_object.close()
    with open(filename, '+a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

#returns the image rotated to level the horizon and the angle of rotation
def rectifyHorizon(img,imgend = None):
    #get only blue channel of the image.  CV imports images as colour order bgr
    blueimg = img[:,:,0]
    width,height = blueimg.shape[:]
    #broadcast the last line of image pixels down the empty timestamp square.  Pure black spaces may affect the thresholding. 
    if imgend is not None:
        lastline = blueimg[imgend,:]
        timestampmask = np.broadcast_to(lastline,(width-imgend,lastline.shape[0]))
        blueimg[imgend:,:] = timestampmask
    
    #x,y cordinates of set of evenly spaced points lying on the edge of the threshold and the binary thresholded image
    x,y,climg = detect_horizon_line(blueimg)
    del blueimg

    #create a line of best fit on the points returns the a and b coefficient of the line formula y=ax+b
    a, b = np.polyfit(x, y, 1)

    liny = a*x+b
    #plt.figure(i)        
    #plt.plot(x,y,"bo")
    #plt.plot(x,liny)

    #Use slope to find the angle of the horizon
    angle = np.arctan(a)*180/np.pi

    #get matrix to rotate image
    rot_matrix = getRotationMatrix(img,angle)
    #get matrix to shift the image vertically
    #vert_matrix = getVerticalShiftMatrix(img,a,b,goldA,goldB)


    #warp image on rotation matrix and then vertical shift matrix
    rot_img = warp(img, rot_matrix)
    # rot_img = warp(rot_img, vert_matrix)
    
    return rot_img,angle
    
#Get exif data from image
def getExif(imgpath):
    im = Image.open(imgpath)
    exif = im.info['exif']
    del im
    return exif

#removes timestamp on image (useful for wingscape images)
def removeImageTimestamp(img, cutoff = 2299):
    img[cutoff:,...] = 0
    return img

#builds a string with attributes joined by '_' for use with file names
def buildName(attributes, suffix):
    return "_".join(attributes) + suffix

#saves image by converting cv2 image to PIL Image object
def saveImg(img,fname,source,keepExif=True):

    #converts to Image type and flips colour channels for saving with PIL
    im = Image.fromarray(img[...,::-1])
    if keepExif:
        exif = getExif(source)
        im.save(fname, exif=exif)
    else:
        im.save(fname)

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

#uses variance of laplace to calculate a blurry index
def isBlurry(img,threshhold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)        
    if fm <= threshhold:
        return False,fm
    else:
        return True,fm

#checks the laplacian variance only at the horizon
#buffer is how far from the horizon line to make the mask
def horizonBlur(img,buffer = 100,imgend = None):
    from horizonFix import detect_horizon_line

    blueimg = img[:,:,0]
    width,height = blueimg.shape[:]
    if imgend is not None:
        #broadcast the last line of image pixels down the empty timestamp square.  Pure black spaces may affect the thresholding. 
        lastline = blueimg[imgend,:]
        timestampmask = np.broadcast_to(lastline,(width-imgend,lastline.shape[0]))
        blueimg[imgend:,:] = timestampmask

    #x,y cordinates of set of evenly spaced points lying on the edge of the threshold and the binary thresholded image
    x,y,climg = detect_horizon_line(blueimg)

    liny_lower =  np.min(y) - buffer
    liny_upper =  np.max(y) + buffer
    imhor = img[liny_lower:liny_upper,:,:]

    _,laplace = isBlurry(imhor,100)
    #free up memory
    del blueimg
    del imhor
    return laplace
