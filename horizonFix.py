
import cv2

from pathlib import Path
import os

from datetime import datetime
from tqdm import tqdm
import re
import pandas as pd
from tools import detect_horizon_line, warp, getRotationMatrix, getVerticalShiftMatrix, getDateTimeFromExif, getDateTimeFromName, writetocsv, rectifyHorizon, getExif, removeImageTimestamp, buildName, saveImg, isBlurry, horizonBlur


if __name__ == '__main__':
    # rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandardRotated_nostamp")
    #rotation_save_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2_rectified")
    # rotation_save_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\rectified_images_exif_test")
    rotation_save_dir = Path(r"/mnt/databackup/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam_test/")
    #rotation_save_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_Phenocam_Futura_Rectified_All_Images_2_leveled/2011")

    # data_dir = Path(r"/mnt/databackup/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Atqasuk_MISP_PhenoCam")
    data_dir = Path(r"/mnt/databackup/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam/")
    # data_dir = Path(r"C:\Users\Daniel\Documents\sel\horizon_rotation_images\goldenStandards")
    #data_dir = Path(r"D:\ITEX-AON_Phenocam_Images\WingScapes_PhenoCam_2011-2015\Utqiagvik_MISP_PhenoCam\2014_2")
    # data_dir = Path(r"/media/dan/dataBackup1/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/Utqiagvik_MISP_PhenoCam/2015/")
    #data_dir = Path(r"/media/dan/ITEX-AON PhenoCam Image MASTER/ITEX-AON_Phenocam_Images/WingScapes_PhenoCam_2011-2015/utqiagvik_MISP_PhenoCam_level/2011/Futura_Rectified_All_Images_2/")

    
    files = list(data_dir.glob("**/*"))
    input_img_paths = [x for x in files if (x.is_file() and ("jpg" in x.suffix.lower() or "jpeg" in x.suffix.lower() or "png" in x.suffix.lower())) ]
    
    timeCorrection = None
    # timeCorrection = readTimeTable("/home/dan/Downloads/2015_Wingscapes_Date_Time_Table.csv")
    

    #this values are the A and B coeffecients of y=Ax+B line for the golden standard image.
    #goldB = 589.2652532259266
    goldB = 582.3542738867756

    #goldA = 0.006334331804510105
    goldA = -0.047456609746488194
    
    siteID = "UTQ"
    
    for imdir in tqdm(input_img_paths):
        try:
            #full file name with suffix
            fullName = Path(imdir).name
            #name without suffix
            name = Path(imdir).stem
            img = cv2.imread(str(imdir))
            
            if img.shape[:2] != (2448, 3264):
                img = cv2.resize(img, (3264,2448))
                print("resizing")
                

            #remove timestamp from image itself
            img = removeImageTimestamp(img)
            
            #get timestamp from exif or datetable if it is from 2015 table
            if timeCorrection is not None  and Path(imdir).parts[-2] == "2015" and timeCorrection['Image'].str.contains(fullName).any():
                datetime = timeCorrection.loc[timeCorrection['Image'] == fullName]["datetime"].item()
            else:
                datetime = getDateTimeFromExif(str(imdir)) 

            year = datetime.strftime("%Y")

            timestamp = datetime.strftime("%Y%m%d%H%M%S")

            # required for old wingscape cameras
            # timestamp = re.sub("^2010","2011",timestamp)
            # year = re.sub("^2010","2011",year)


            rot_img,angle = rectifyHorizon(img,imgend = 2299)                        
            
            #Store image seperated by year
            finalDir = (rotation_save_dir / year) #getfilestructure(imdir))
            os.makedirs(finalDir,exist_ok=True)




            rotname = buildName([timestamp, siteID, name, "leveled"], ".jpg")

            rotname = re.sub("^2010","2011",rotname)



            saveImg(rot_img,str(finalDir/rotname),imdir)

            #create csv of the stats
            _,laplace = isBlurry(img,0)
            # try:
            horilaplace = horizonBlur(img,imgend = 2299)
            # except Exception as e:
                # raise e

            row = [rotname,timestamp,angle,laplace, horilaplace]

            csvName = buildName([siteID,"level",year,"stats","horizonBlur"],".csv")
            writetocsv(finalDir/csvName,row,title=["filename","timestamp","horizonAngle","laplacianVariance","laplacianVarianceHorizon"])
            del rot_img
        except Exception as e:
            print(e)
            print("could not level",name)
            print("source directory",imdir)
        