"""Utilities for finding the horizon line"""

import cv2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

#last line of the image before timestamp
imgend = 2299


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

    x = np.arange(horizon_x1,horizon_x2,10)
    y = []
    for i in x:
        y.append(min(np.where(image_closed[:, i] == 255)[0]))
    #cv2.imshow("hi",image_closed)
    #return image_closed
    return x,y,image_closed
    return horizon_x1, horizon_x2, horizon_y1, horizon_y2


if __name__ == '__main__':
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated")
    green_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\green_channel")
    horizon_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\horizon_detection")

    data_dir = Path(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification")
    data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")

    input_img_paths = sorted(
    [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".JPG") or fname.endswith(".jpg")
    ]
    )
    
    print(input_img_paths)
    
    i = 1
    for imdir in input_img_paths:
        # img = cv2.imread(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification\WSBC0029.JPG",cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(imdir,cv2.IMREAD_GRAYSCALE)
        name = Path(imdir).stem
        img = cv2.imread(imdir)
        grnimg = img[:,:,0]
        width,height = grnimg.shape[:]

        
        lastline = grnimg[imgend,:]
        timestampmask = np.broadcast_to(lastline,(width-imgend,lastline.shape[0]))
        grnimg[2299:,:] = timestampmask
        
        x,y,climg = detect_horizon_line(grnimg)
        a, b = np.polyfit(x, y, 1)

        plt.figure(i)
        
        plt.plot(x,y,"bo")
        liny = a*x+b
        plt.plot(x,liny)

        #angle = np.arctan((max(liny)-min(liny))/max(x))*180/np.pi#np.arctan(-a)*2*np.pi
        angle = np.arctan(a)*180/np.pi
        print("angle",angle)
        
        plt.imshow(climg)
        horzname = name + "_" + "horizon" + ".jpg"
        # cv2.imwrite(str(horizon_save_dir/horzname),climg)
        #plt.savefig(str(horizon_save_dir/horzname))


        grnname = name + "_" + "green" + ".jpg"
        #cv2.imwrite(str(green_save_dir/grnname),grnimg)
        
        
        center = (width//2, height//2)
        matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rot_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        plt.figure(i+1)
        i+=2
        # plt.imshow(rot_img)
        
        rotname = name + "_" + str(round(angle,4)) + ".jpg"
        #cv2.imwrite(str(rotation_save_dir/rotname),rot_img)
        
        # plt.imsave("")