
import cv2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from PIL import Image, ExifTags

from datetime import datetime
from tqdm import tqdm

#last line of the image before timestamp
imgend = 2299
#goldB = 589.2652532259266
goldB = 582.3542738867756

#goldA = 0.006334331804510105
goldA = -0.047456609746488194


#inspired by method developed by Sean Sall origin: https://github.com/sallamander/horizon-detection/blob/master/utils.py
def detect_horizon_line(image_grayscaled):
    """Detect the horizon's starting and ending points in the given image

    The horizon line is detected by applying Otsu's threshold method to
    separate the sky from the remainder of the image.

    :param image_grayscaled: grayscaled image to detect the horizon on, of
     shape (height, width)
    :type image_grayscale: np.ndarray of dtype uint8
    :return: the (x1, x2, y1, y2) coordinates for the starting and ending
     points of the detected horizon line
    :rtype: tuple(int)
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
def getDateTime(imdir):
    pimg = Image.open(str(imdir))
    img_exif = pimg.getexif()
    #this is the code for datetime
    strdatetime = img_exif[306]
    timestamp = datetime.strptime(strdatetime,"%Y:%m:%d %H:%M:%S")
    return timestamp

#returns directory structure with parent directories removed
def getfilestructure(path):
    return str(path.parent).replace(str(data_dir),"")[1:] #indexing to remove starting /

def writetocsv(filename,row):
    import csv
    with open(filename, '+a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def isBlurry(img,threshhold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)        
    if fm <= threshhold:
        return False,fm
    else:
        return True,fm



    

if __name__ == '__main__':
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated_nostamp")
    #rotation_save_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2_rectified")
    #rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\rectified_images_nostamp")
    rotation_save_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_PhenoCam_level/")
    rotation_save_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_Phenocam_Futura_Rectified_All_Images_2_leveled/2011/")
    
    green_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\green_channel")
    horizon_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\horizon_detection")

    #data_dir = Path(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification")
    data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")
    #data_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2")
    data_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam/")
    data_dir = Path(r"/media/dan/dataBackup1/phenocam2016-2021/2016_Brw_MISP_RGB_Flattened/")

    
    files = list(data_dir.glob("**/*"))
    input_img_paths = [x for x in files if (x.is_file() and "jpeg" in x.suffix.lower()) ]


    
    
    i = 1
    blurs = []
    for imdir in tqdm(input_img_paths):
        name = Path(imdir).stem
        img = cv2.imread(str(imdir))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm <= 25:
            text = "Blurry"
        # show the image
        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imwrite("blurrytest/blur_"+imdir.name,img)
        blurs.append(fm)
        #cv2.imshow("Image", img)
        #key = cv2.waitKey(0)        
            

        