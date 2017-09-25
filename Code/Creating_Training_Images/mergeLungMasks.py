import glob
import cv2
import os

def main():
    leftMasks = glob.glob('d:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\LeftLung\\*')
    rightMasks = glob.glob('d:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\RightLung\\*')

    for (left, right) in zip(leftMasks, rightMasks):
        leftImage = cv2.imread(left)
        rightImage = cv2.imread(right)
        merged = cv2.bitwise_or(leftImage, rightImage)

        _, fileName = os.path.split(left)
        cv2.imwrite("d:\\School\\2017Onlab1\\Code\\Creating_Training_Images\\EveryImage\\LungMasks\\" + fileName, merged)

if __name__ == '__main__':
    main()
    print("Done processing")