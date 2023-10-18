"""Utilities for finding the horizon line"""

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

def warp(img,matrix):
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

def getRotationMatrix(img,angle):
    width,height = img.shape[:2]
    center = (width//2, height//2)
    return cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

def getVerticalShiftMatrix(img,a,b,goldA,goldB):
    width,height = img.shape[:2]
    center = (width//2, height//2)
    
    horizonHeight = a*center[1] + b
    goldHorizonHeight = goldA*center[1] + goldB
    
    diff = goldHorizonHeight - horizonHeight
    return np.array([[1,0,0],
                     [0,1,diff]])

    

#datetime.isoformat()
def getDateTime(imdir):
    pimg = Image.open(str(imdir))
    img_exif = pimg.getexif()
    #this is the code for datetime
    strdatetime = img_exif[306]
    timestamp = datetime.strptime(strdatetime,"%Y:%m:%d %H:%M:%S")
    return timestamp

def getkeydescSift(img1,img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    return kp1, des1, kp2, des2

def flannMatcher(kp1,des1,kp2,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    return matches

def drawgraph(img1,img2,kp1,des1,kp2,des2,matches):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
     if m.distance < 0.7*n.distance:
         matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
    singlePointColor = (255,0,0),
    matchesMask = matchesMask,
    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.USAC_MAGSAC ,5)
    choice = np.where(mask.reshape(-1) ==1)[0]
    return pts0[choice], pts1[choice]

def getPoints(img1, img2):
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img1,mask=None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2,mask=None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1,descriptors2)
    # Extract data from orb objects and matcher
    dist = np.array([m.distance for m in matches])
    ind1 = np.array([m.queryIdx for m in matches])
    ind2 = np.array([m.trainIdx for m in matches])
    keypoints1 = np.array([p.pt for p in keypoints1])
    keypoints2 = np.array([p.pt for p in keypoints2])
    keypoints1 = keypoints1[ind1]
    keypoints2 = keypoints2[ind2]
    keypoints1, keypoints2 = select_matches_ransac(keypoints1, keypoints2)

    return keypoints1, keypoints2

def getPointsMasked(img1, img2,common_mask=None):
    orb = cv2.ORB_create()

    x = np.arange(0,common_mask.shape[1],10)
    bottom = 0
    img1 = cv2.GaussianBlur(img1, ksize=(3, 3), sigmaX=0)
    img2 = cv2.GaussianBlur(img2, ksize=(3, 3), sigmaX=0)
    for i in x:
        lowerend = np.where(common_mask[:, i] == True)[0]
        if lowerend.shape[0] > 0:
            new_bottom = max(lowerend)
            if bottom < new_bottom:
                bottom = new_bottom
                
    band = 200
    keypoints1 = keypoints2  = None
    descriptors1 = descriptors2 = None
    
    keys1 = []
    keys2 = []
    hight, width = img1.shape[:2]
    center = (hight//2, width//2)

    goldHorizonHeight = int(goldA*center[1] + goldB)

    for i in range(bottom,goldHorizonHeight,-band):
        band_mask = np.zeros_like(common_mask,dtype=np.uint8)
        band_mask[i:i+band,:] = 1
        mask = common_mask * band_mask

        keypoints1, descriptors1 = orb.detectAndCompute(img1*mask,mask=None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2*mask,mask=None)
        #print(len(k1),len(d1))
        #print(len(k2),len(d2))
        # if keypoints1 is None:
        #     keypoints1 = k1
        # elif k1 is not None and keypoints1 is not None:
        #     keypoints1 +=k1
        # if keypoints2 is None:
        #     keypoints2 = k2
        # elif k2 is not None and keypoints2 is not None:
        #     keypoints2 +=k2
        # #descriptors1 += d1
        # #descriptors2 += d2
        # if descriptors1 is None:
        #     descriptors1 = d1
        # elif d1 is not None:

        #     descriptors1 = np.concatenate([descriptors1,d1])
        # if descriptors2 is None:
        #     descriptors2 = d2
        # elif d2 is not None:
        #     descriptors2 = np.concatenate([descriptors2,d2])
        
        if keypoints1 is not None and keypoints2 is not None and descriptors1 is not None and descriptors2 is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(descriptors1,descriptors2)
            # Extract data from orb objects and matcher
            dist = np.array([m.distance for m in matches])
            ind1 = np.array([m.queryIdx for m in matches])
            ind2 = np.array([m.trainIdx for m in matches])
            keypoints1 = np.array([p.pt for p in keypoints1])
            keypoints2 = np.array([p.pt for p in keypoints2])
            keypoints1 = keypoints1[ind1]
            keypoints2 = keypoints2[ind2]
            #keypoints1, keypoints2 = select_matches_ransac(keypoints1, keypoints2)
            keys1.append(keypoints1)
            keys2.append(keypoints2)
    keypoints1,keypoints2 = np.concatenate(keys1), np.concatenate(keys2)
    keypoints1, keypoints2 = select_matches_ransac(keypoints1, keypoints2)

    return keypoints1,keypoints2

def display_control_lines(im0,im1,pts0=np.array([[0,0]]),pts1=np.array([[0,0]]),clr_str = 'rgbycmwk',tag=""):
    canvas_shape = (max(im0.shape[0],im1.shape[0]),im0.shape[1]+im1.shape[1],3)
    canvas = np.zeros(canvas_shape,dtype=type(im0[0,0,0]))
    canvas[:im0.shape[0],:im0.shape[1]] = im0
    canvas[:im1.shape[0],im0.shape[1]:canvas.shape[1]]= im1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(canvas)
    ax.axis('off')
    pts2 = pts1+np.array([im0.shape[1],0])
    for i in range(pts0.shape[0]):
        ax.plot([pts0[i,0],pts2[i,0]],[pts0[i,1],pts2[i,1]],color=clr_str[i%len(clr_str)],linewidth=1.0)
    fig.suptitle('Point correpondences', fontsize=16)
    

if __name__ == '__main__':
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated")
    rotation_save_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2_rectified")
    rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\registered_images")
    
    green_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\green_channel")
    horizon_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\horizon_detection")

    data_dir = Path(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification")
    #data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")
    #data_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2")

    # input_img_paths = sorted(
    # [
    #     os.path.join(data_dir, fname)
    #     for fname in os.listdir(data_dir)
    #     if fname.endswith(".JPG") or fname.endswith(".jpg")
    # ]
    # )
    
    files = list(data_dir.glob("**/*"))
    input_img_paths = [x for x in files if (x.is_file() and "JPG" in x.suffix) ]


    
    gsDir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated_nostamp\goldenStandard_-2.717_20130810161302.jpg")
    gsimg = cv2.imread(str(gsDir))
    
    i = 1
    for imdir in tqdm(input_img_paths[:]):
        # img = cv2.imread(r"C:\Users\Daniel\Documents\sel\Example_Images_For_Rectification\WSBC0029.JPG",cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(imdir,cv2.IMREAD_GRAYSCALE)
        name = Path(imdir).stem
        img = cv2.imread(str(imdir))
        #remote timestamp
        img[2299:,...] = 0
        
        timestamp = getDateTime(str(imdir))
        
        grnimg = img[:,:,0]
        width,height = grnimg.shape[:]

        
        lastline = grnimg[imgend,:]
        timestampmask = np.broadcast_to(lastline,(width-imgend,lastline.shape[0]))
        grnimg[2299:,:] = timestampmask
        
        x,y,climg = detect_horizon_line(grnimg)
        del grnimg
        a, b = np.polyfit(x, y, 1)

        #plt.figure(i)
        
        #plt.plot(x,y,"bo")
        liny = a*x+b
        #plt.plot(x,liny)

        #angle = np.arctan((max(liny)-min(liny))/max(x))*180/np.pi#np.arctan(-a)*2*np.pi
        angle = np.arctan(a)*180/np.pi
        #print("angle",angle)
        
        #plt.imshow(climg)
        horzname = name + "_" + "horizon" + ".jpg"
        # cv2.imwrite(str(horizon_save_dir/horzname),climg)
        #plt.savefig(str(horizon_save_dir/horzname))


        grnname = name + "_" + "green" + ".jpg"
        #cv2.imwrite(str(green_save_dir/grnname),grnimg)
        
        rot_matrix = getRotationMatrix(img,angle)
        vert_matrix = getVerticalShiftMatrix(img,a,b,goldA,goldB)
        matrix = vert_matrix * rot_matrix  


        rot_img = warp(img, rot_matrix)
        rot_img = warp(rot_img, vert_matrix)
        
        #plt.figure(i+1)
        i+=2
        #plt.imshow(rot_img)
        
        rotname = name + "_" + str(round(angle,4)) +"_" + timestamp.strftime("%Y%m%d%H%M%S") + ".jpg"
        #cv2.imwrite(str(rotation_save_dir/rotname),rot_img)
        #del rot_img

        gsimg_gray = cv2.cvtColor(gsimg, cv2.COLOR_BGR2GRAY) 
        img_gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        gsimg_mask = gsimg_gray > 0
        img_mask = img_gray > 0
        
        common_mask = gsimg_mask * img_mask
        
        gsimg_m = common_mask * gsimg_gray
        img_m = common_mask * img_gray
        
        
        
        key1, key2 = getPointsMasked(gsimg_m, img_m,common_mask)
        #keydiff = (key1-key2).mean(axis=0)
        #lateraldiff = keydiff[0]
        e = (key1-key2)[:,1].argsort()
        lateraldiff = (key1-key2)[e][0][0]
        #lateraldiff = np.median((key1-key2)[0])
        shift_mat = np.array([[1,0,lateraldiff],
                         [0,1,0]],dtype=np.float32)
        rot_shift_img = warp(rot_img,shift_mat)
        
        #display_control_lines(gsimg[...,::-1],rot_img[...,::-1],key1, key2)
        cv2.imwrite(str(rotation_save_dir/rotname),rot_shift_img)

    #plt.figure(2)
    #plt.imshow(gsimg)
    #plt.figure(3)
    #plt.imshow(rot_shift_img)

        # plt.imsave("")