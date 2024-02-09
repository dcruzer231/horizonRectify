
import cv2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from PIL import Image, ExifTags

from datetime import datetime
from tqdm import tqdm
import re
from blurIndex import isBlurry
import pandas as pd




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
def getDateTimeFromExif(imdir):
    pimg = Image.open(str(imdir))
    img_exif = pimg.getexif()
    #this is the code for datetime
    strdatetime = img_exif[306]
    timestamp = datetime.strptime(strdatetime,"%Y:%m:%d %H:%M:%S")
    return timestamp

#returns directory structure with parent directories removed
def getfilestructure(path):
    return str(path.parent).replace(str(data_dir),"")[1:] #indexing to remove starting /

#writes a single row to a csv file, row is expected to be a list
def writetocsv(filename,row):
    import csv
    with open(filename, '+a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

#returns the image rotated to level the horizon and the angle of rotation
def rectifyHorizon(img,imgend = 2299):
    #get only blue channel of the image
    blueimg = img[:,:,0]
    width,height = blueimg.shape[:]
    #broadcast the last line of image pixels down the empty timestamp square.  Pure black spaces may affect the thresholding. 
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

#removes timestamp on image (usefule for wingscape images)
def removeImageTimestamp(img, cutoff = 2299):
    img[cutoff:,...] = 0
    return img

#builds a string with attributes joined by '_' for use with file names
def buildName(attributes, suffix):
    return "_".join(attributes) + suffix

#saves image by converting cv2 image to PIL Image object
def saveImg(img,fname,keepExif=True):

    #converts to Image type and flips colour channels
    im = Image.fromarray(img[...,::-1])
    if keepExif:
        exif = getExif(imdir)
        im.save(fname, exif=exif)
    else:
        im.save(fname)


if __name__ == '__main__':
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated_nostamp")
    #rotation_save_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2_rectified")
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\rectified_images_exif_test")
    # rotation_save_dir = Path(r"/media/dan/dataBackup1/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_PhenoCam_level/")
    #rotation_save_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_Phenocam_Futura_Rectified_All_Images_2_leveled/2011")

    data_dir = Path(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification")
    # data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")
    #data_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2")
    # data_dir = Path(r"/media/dan/dataBackup1/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam/2015/")
    #data_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_PhenoCam_level/2011/Futura_Rectified_All_Images_2/")

    
    files = list(data_dir.glob("**/*"))
    input_img_paths = [x for x in files if (x.is_file() and ("jpg" in x.suffix.lower() or "jpeg" in x.suffix.lower() or "png" in x.suffix.lower())) ]
    
    # timeCorrection = pd.read_csv("/home/dan/Downloads/2015_Wingscapes_Date_Time_Table.csv")
    # timeCorrection["datetime"] = timeCorrection["Date"] + " " + timeCorrection["Time"]
    
    # timeCorrection["datetime"] = pd.to_datetime(timeCorrection["datetime"],format="%m/%d/%Y %I:%M %p")
    #last line of the image before timestamp
    

    #this values are the A and B coeffecients of y=Ax+B line for the golden standard image.
    #goldB = 589.2652532259266
    goldB = 582.3542738867756

    #goldA = 0.006334331804510105
    goldA = -0.047456609746488194
    
    tag = "UTQ"
    
    for imdir in tqdm(input_img_paths):
        try:
            
            name = Path(imdir).stem
            img = cv2.imread(str(imdir))
            
            if img.shape[:2] != (2448, 3264):
                img = cv2.resize(img, (3264,2448))
                print("resizing")
                

            #remove timestamp from image itself
            img = removeImageTimestamp(img)
            
            #get timestamp from exif
            datetime = getDateTimeFromExif(str(imdir)) #timeCorrection.loc[timeCorrection['Image'] == name+".JPG"]["datetime"].item()

            timestamp = datetime.strftime("%Y%m%d%H%M%S")
            year = timestamp.strftime("%Y")

            # required for old wingscape cameras
            # timestamp = re.sub("^2010","2011",timestamp)
            # year = re.sub("^2010","2011",year)


            rot_img,angle = rectifyHorizon(img)                        
            
            #Store image seperated by year
            finalDir = (rotation_save_dir / year) #getfilestructure(imdir))
            os.makedirs(finalDir,exist_ok=True)




            rotname = buildName([timestamp, tag, name, "leveled"], ".jpg")

            rotname = re.sub("^2010","2011",rotname)



            saveImg(rot_img,str(finalDir/rotname))

            #creat csv of the stats
            _,laplace = isBlurry(img,0)
            row = [rotname,timestamp,angle,laplace]

            csvName = buildName([tag,"level","2015","stats"],".csv")
            writetocsv(csvName,row)
            del rot_img
        except Exception as e:
            print(e)
            print("could not level",name)
            print("source directory",imdir)
        