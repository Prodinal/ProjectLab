import glob
import os
from PIL import Image


def convertImageToPng(imageFile):
    filepath,filename = os.path.split(imageFile)
    filterame,exts = os.path.splitext(filename)
    print("Processing: " + imageFile,filterame)
    return Image.open(imageFile), filterame
    
def main():
    files = glob.glob("d:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\LeftLungGif\\*.gif") 

    for imageName in files:
        im, filterName = convertImageToPng(imageName)
        im.save( 'd:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\LeftLung\\'+filterName+'.png','PNG')

    files = glob.glob("d:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\RightLungGif\\*.gif")

    for imageName in files:
        im, filterName = convertImageToPng(imageName)
        im.save( 'd:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\RightLung\\'+filterName+'.png','PNG')

if __name__ == '__main__':
    main()
    print("Done processing")