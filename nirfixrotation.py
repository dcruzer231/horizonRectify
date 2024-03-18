"""
This program is used to fix the rotation of IR pictures by comparing it to the nearest RGB twin image.
"""

import pandas as pd
from horizonFix import *
from pathlib import Path
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    dataPath = Path("/mnt/databackup/ITEX-AON_Phenocam_Images/Phenocam_2016-2022/Atqasuk/phenocam_ATQ_time_corrected/2019/RC0004/NIR/")
    rotation_save_dir = Path("/mnt/databackup/ITEX-AON_Phenocam_Images/Phenocam_2016-2022/Atqasuk/Phenocam_ATQ_NIR_RGBAligned/")

    df = pd.read_csv("/mnt/databackup/ITEX-AON_Phenocam_Images/Phenocam_2016-2022/Atqasuk/Phenocam_ATQ_Leveled/2019/RC0004/RGB/ATQ_level_RC0004_RGB_2019_stats.csv")

    tag = "ATQ"

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y%m%d%H%M%S")
    nirPaths = list(dataPath.glob("**/*"))
    avgImg = None
    for imgPath in tqdm(nirPaths):
        name = imgPath.stem
        nirDatetime = getDateTimeFromName(name)

        img = cv2.imread(str(imgPath))

        #extract serial from path.  
        for part in Path(imgPath).parts[:-1]:
            if "RC" in part:
                serial = part
            if "rgb" or "ir" in part.lower():
                imgType = part
        

        #create csv of the stats
        _,laplace = isBlurry(img,0)
        
        #Find the closest timestamp to the NIR timestamp
        df["timeDiff"] = abs(df.timestamp - nirDatetime)
        closest_row = df.loc[df['timeDiff'].idxmin()]
        angle = closest_row.horizonAngle

        year = nirDatetime.strftime("%Y")
        timestamp = nirDatetime.strftime("%Y%m%d%H%M%S")

        #Store image seperated by year
        finalDir = (rotation_save_dir / year / serial / imgType) #getfilestructure(imdir))
        os.makedirs(finalDir,exist_ok=True)

        if pd.isna(angle):
            rotname = buildName([tag, name, "notleveled"], ".jpg")
            saveImg(img,str(finalDir/rotname),imgPath)
            row = [rotname,timestamp,None,laplace]
            csvName = buildName([tag,"level", serial, imgType, year,"stats","RGBAligned"],".csv")
            writetocsv(finalDir/csvName,row,title=["filename","timestamp","horizonAngle","laplacianVariance"])

        rot_matrix = getRotationMatrix(img,angle)
        rot_img = warp(img, rot_matrix)

        rotname = buildName([tag, name, "leveled"], ".jpg")
        saveImg(rot_img,str(finalDir/rotname),imgPath)

        row = [rotname,timestamp,angle,laplace]
        csvName = buildName([tag,"level", serial, imgType, year,"stats","RGBAligned"],".csv")
        writetocsv(finalDir/csvName,row,title=["filename","timestamp","horizonAngle","laplacianVariance"])

