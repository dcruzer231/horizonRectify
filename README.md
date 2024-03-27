# Scripts and tools to rectify horizon
## Contents
- conda_requirements.txt - a list of requirements for building a conda environemnt to run these scripts.  
- featureDetection.py - legacy code, previous attempts to try to rectify images based off of feature detection between a golden standard image.  Script is left here to catalogue previous attempts.
- horizonFix_wingscape.py - main script to horizon rectify wingscape images.
- horizonFix_phenocam.py - main script to horizon rectify phenocam images.
- nirfixrotation.py - script to rotate phenocam NIR images based off the RGB counterpart.  This was done because the horizon rectification strategy does not work well on NIR images.
- tools.py - holds all the methods used for horizon rectification, making csv files, saving images, etc.
## Current capabilities
- detect horizon line and level image
- save image with a new name
- calculate a 'blurriness' index for an image
- calculate an index for fog in an image based on horizon blurriness
- fix date errors in wingscape images
## setup
This project used conda environments to set up the module dependencies.  To create an environment called horizonRectify based off of requirements:
```bash
conda create --name horizonRectify
conda activate horizonRectify
conda install --file conda_requirements.txt
'''
