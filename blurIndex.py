
import cv2
import numpy as np
from pathlib import Path
from tools import variance_of_laplacian, isBlurry, horizonBlur


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
        from tqdm import tqdm
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
            

        