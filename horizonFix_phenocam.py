"""
Script to horizon rectify and calculate blur on phenocam images
"""


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
from tools import isBlurry, horizonBlur, removeImageTimestamp, getDateTimeFromName, writetocsv, buildName, saveImg, rectifyHorizon

if __name__ == '__main__':
    # rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated_nostamp")
    #rotation_save_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2_rectified")
    # rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\rectified_images_exif_test")
    rotation_save_dir = Path(r"/mnt/databackup/ITEX-AON_Phenocam_Images/Phenocam_2016-2022/Utqiagvik/Phenocam_UTQ_test/")
    #rotation_save_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_Phenocam_Futura_Rectified_All_Images_2_leveled/2011")

    data_dir = Path(r"/mnt/databackup/ITEX-AON_Phenocam_Images/Phenocam_2016-2022/Utqiagvik/Phenocam_correctedDate2/")
    # data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")
    #data_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2")
    # data_dir = Path(r"/media/dan/dataBackup1/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam/2015/")
    #data_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_PhenoCam_level/2011/Futura_Rectified_All_Images_2/")

    
    files = list(data_dir.glob("**/*"))
    input_img_paths = [x for x in files if (x.is_file() and ("jpg" in x.suffix.lower() or "jpeg" in x.suffix.lower() or "png" in x.suffix.lower())) ]
    
    timeCorrection = None
    # timeCorrection = readTimeTable("/home/dan/Downloads/2015_Wingscapes_Date_Time_Table.csv")
    

    
    siteID = "UTQ"
    
    for imdir in tqdm(input_img_paths):
        if "NIR" in str(imdir):
            continue
        try:
            #full file name with suffix
            fullName = Path(imdir).name
            #name without suffix
            name = Path(imdir).stem
            img = cv2.imread(str(imdir))

            #extract serial from path.  
            for part in Path(imdir).parts[:-1]:
                if "RC" in part:
                    serial = part
                if "rgb" or "ir" in part.lower():
                    imgType = part
                
            
            #get timestamp from exif or datetable if it is from 2015 wingscape table
            if timeCorrection is not None  and Path(imdir).parts[-2] == "2015" and timeCorrection['Image'].str.contains(fullName).any():
                datetime = timeCorrection.loc[timeCorrection['Image'] == fullName]["datetime"].item()
            else:
                datetime = getDateTimeFromName(name) 

            year = datetime.strftime("%Y")

            timestamp = datetime.strftime("%Y%m%d%H%M%S")

            #create csv of the stats
            _,laplace = isBlurry(img,0)


            #Store image seperated by year
            finalDir = (rotation_save_dir / year / serial / imgType)
            os.makedirs(finalDir,exist_ok=True)


            horilaplace = horizonBlur(img)
            rot_img,angle = rectifyHorizon(img)                        
            


            rotname = buildName([siteID, name, "leveled"], ".jpg")



            saveImg(rot_img,str(finalDir/rotname),imdir)


            row = [rotname,timestamp,angle,laplace,horilaplace]

            csvName = buildName([siteID,"level", serial, imgType, year,"stats","horizonBlur"],".csv")
            writetocsv(finalDir/csvName,row,title=["filename","timestamp","horizonAngle","laplacianVariance","laplacianVarianceHorizon"])
            del rot_img
        except Exception as e:
            print(e)
            print("could not level",name)
            print("source directory",imdir)

            rotname = buildName([siteID, name, "notleveled"], ".jpg")
            # saveImg(img,str(finalDir/rotname),imdir)
            row = [rotname,timestamp,None,laplace,None]
            csvName = buildName([siteID,"level", serial, imgType, year,"stats","horizonBlur"],".csv")
            writetocsv(finalDir/csvName,row,title=["filename","timestamp","horizonAngle","laplacianVariance","laplacianVarianceHorizon"])

        