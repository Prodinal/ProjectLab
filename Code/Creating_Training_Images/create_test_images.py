import random
import os
import sys
import glob
import numpy as np
import cv2
import commonImageTrain as common

#Global variables
oper = ""
enhanced = ""
roiSize = ""

#Constants
validationImagesPath = os.path.dirname(__file__) + "\\ValidateImages\\WholeImages\\"
validationImageSlicesPath = os.path.dirname(__file__) + "\\ValidateImages\\ImageSlices\\"

negativeImagePerPositive = 15


# Initialize global variables
if __name__ == "__main__":
    if(len(sys.argv) < 4):
        sys.exit(-1)

    oper = sys.argv[2]
    if oper == 'nodule' or oper == 'Nodule':
        operation = 'nodule'
    elif oper == 'highlight' or oper == 'Highlight':
        operation = 'highlight'
    else:
        oper = 'nodule'

    enhanced = sys.argv[3]
    if enhanced.lower() == 'yes':
        fileNameExtension = 'Result'
    else:
        fileNameExtension = ''
    roiSize = int(int(sys.argv[1]) / 2)



def getFileNameFromLine(line):
    parts = line.split('	')
    return parts[0]


def getCoordsFromLine(line):
    parts = line.split('	')
    return int(parts[5]), int(parts[6])

def getCorrectCoords(file, name):
    for line in file:
        if getFileNameFromLine(line).split('.')[0] == name.replace('Result', '').split('.')[0]:
            return getCoordsFromLine(line)


def deleteOldFiles():
    def deleteFiles(deleteFiles):
        for f in deleteFiles:
            # print(f)
            os.remove(f)

    deleteFiles(glob.glob(common.images_root_path + "\\Nodules\\negative\\*"))
    deleteFiles(glob.glob(common.images_root_path + "\\Nodules\\positive\\*"))
    deleteFiles(glob.glob(common.images_root_path + "\\Highlights\\positive\\*"))
    deleteFiles(glob.glob(common.images_root_path + "\\Highlights\\negative\\*"))
    deleteFiles(glob.glob(common.images_root_path + "\\Highlights\\masks\\*"))
    deleteFiles(glob.glob(validationImageSlicesPath + "*"))


def openMaskByName(fileName):
    path = os.path.join(common.images_root_path,
                        "LungMasks",
                        fileName)
    return cv2.imread(path)

def openImageByName(fileName):
    path = os.path.join(common.images_root_path,
                        fileName)
    return cv2.imread(path)


def saveImageByName(fileName, image, folder='Nodules', positive='positive'):
    path = os.path.join(common.images_root_path,
                        folder,
                        positive,
                        fileName)
    cv2.imwrite(path, image)
    # print('Done creating image: ' + path)


def createResultImageName(originalName):
    newName = originalName.split('.')[0]
    newName += 'Nodule' + str(roiSize * 2) + '.jpg'
    return newName


def highlightNodule(source, xCoord, yCoord, size=roiSize):
    result = source.copy()
    bottomLeftCoords = (xCoord-size, yCoord-size)
    topRightCoords = (xCoord+size, yCoord+size)
    cv2.rectangle(result,
                  bottomLeftCoords,
                  topRightCoords,
                  (0, 0, 255),
                  2)
    return result


def getRoi(source, xCoord, yCoord, size=roiSize):
    return source[yCoord-size: yCoord+size,
                  xCoord-size: xCoord+size]


def showResizedImage(name, image, fx, fy):
    # cv2.imshow(image)
    cv2.imshow(str(name), cv2.resize(image, (0, 0), fx=fx, fy=fy))


def main():
    deleteOldFiles()
    createLearningImages()
    createValidatingImages()


def createValidatingImages():
    validationImages = glob.glob(os.path.join(os.path.dirname(__file__), "ValidateImages\\WholeImages\\*Result*"))
    f = open(os.path.join(os.path.dirname(__file__),
                          'EveryImage\\CLNDAT_E.txt'),
             'r')
    savePath = validationImageSlicesPath
    for filePath in validationImages:
        imageName = os.path.basename(filePath)
        imageBaseName = imageName.split('.')[0]
        x, y = getCorrectCoords(f, imageName)
        image_data = cv2.imread(filePath)
        rows, cols, _ = image_data.shape
        center = (int(rows/2), int(cols/2))

        # create the 4 flipped positive images
        noduleRoi0 = getRoi(image_data, x, y)
        cv2.imwrite(savePath + imageBaseName + "Positive0.png", noduleRoi0)

        noduleRoi90 = np.rot90(noduleRoi0)
        cv2.imwrite(savePath + imageBaseName + "Positive90.png", noduleRoi90)

        noduleRoi180 = np.rot90(noduleRoi90)
        cv2.imwrite(savePath + imageBaseName + "Positive180.png", noduleRoi180)

        noduleRoi270 = np.rot90(noduleRoi180)
        cv2.imwrite(savePath + imageBaseName + "Positive270.png", noduleRoi270)


        # create negative validation images
        def good_negative_position():
                crit = []
                # size has to be roiSize/4 as this is the size in each direction
                maskRoi = getRoi(lungMask, int(negativeX / 2), int(negativeY / 2), int(roiSize/4))
                dividedImage = np.divide(np.asarray(maskRoi), 255)
                roiArea = int(roiSize/2) * int(roiSize/2)

                crit.append(np.sum(dividedImage)/roiArea > 0.8) #the lung area is at least 80% of the image
                crit.append((lungMask[int(negativeX / 2), int(negativeY / 2)] == 255).all())
                crit.append(abs(negativeX - x) >= roiSize)
                crit.append(abs(negativeY - y) >= roiSize)
                return np.asarray(crit).all()
        

        lungMask = openMaskByName(imageBaseName.replace("Result", '') + ".png")

        for i in range(10):
            negativeX = random.randrange(roiSize, rows-roiSize)
            negativeY = random.randrange(roiSize, cols-roiSize)
            while not good_negative_position():
                negativeX = random.randrange(roiSize, rows-roiSize)
                negativeY = random.randrange(roiSize, cols-roiSize)
            
            negativeRoi = getRoi(image_data, negativeX, negativeY)
            cv2.imwrite(savePath + imageBaseName + "Negative" + str(i) + ".png", negativeRoi)

def createLearningImages():
    skipped_images = []
    f = open(os.path.join(os.path.dirname(__file__),
                          'EveryImage\\CLNDAT_E.txt'),
             'r')

    i = 0
    for line in f:
        xCoord, yCoord = getCoordsFromLine(line)
        xCoord = int(xCoord)
        yCoord = int(yCoord)
        fileName = getFileNameFromLine(line)
        fileName = fileName.split('.')[0]
        sourceImage = openImageByName(fileName + fileNameExtension + '.png')

        lungMask = openMaskByName(fileName + ".png")
        lungMask = cv2.cvtColor(lungMask, cv2.COLOR_BGR2GRAY)

        # some images have not been enhanced, i. e. JPCLN70
        if sourceImage is None:
            skipped_images.append(fileName)
            continue

        saveName = createResultImageName(fileName)

        if operation == 'nodule':
            roi = getRoi(sourceImage, xCoord, yCoord)
            saveImageByName(saveName, roi, 'Nodules', 'positive')
            # saveImageByName(saveName.split('.')[0] + "fliplr.jpg", np.fliplr(roi), 'Nodules', 'positive')
            # saveImageByName(saveName.split('.')[0] + "flipud.jpg", np.flipud(roi), 'Nodules', 'positive')
            saveImageByName(saveName.split('.')[0] + "90.jpg", np.rot90(roi, 1), 'Nodules', 'positive')
            saveImageByName(saveName.split('.')[0] + "180.jpg", np.rot90(roi, 2), 'Nodules', 'positive')
            saveImageByName(saveName.split('.')[0] + "270.jpg", np.rot90(roi, 3), 'Nodules', 'positive')
            highlighted = highlightNodule(sourceImage, xCoord, yCoord)
            saveImageByName(saveName, highlighted, 'Highlights', 'positive')

            def good_negative_position():
                crit = []
                # size has to be roiSize/4 as this is the size in each direction
                maskRoi = getRoi(lungMask, int(negativeX / 2), int(negativeY / 2), int(roiSize/4))
                dividedImage = np.divide(np.asarray(maskRoi), 255)
                roiArea = int(roiSize/2) * int(roiSize/2)

                crit.append(np.sum(dividedImage)/roiArea > 0.8) #the lung area is at least 80% of the image
                crit.append((lungMask[int(negativeX / 2), int(negativeY / 2)] == 255).all())
                crit.append(abs(negativeX - xCoord) >= roiSize)
                crit.append(abs(negativeY - yCoord) >= roiSize)
                return np.asarray(crit).all()

            height, width, channels = sourceImage.shape
            
            for i in range(negativeImagePerPositive):
                negativeX = random.randrange(roiSize, width-roiSize)
                negativeY = random.randrange(roiSize, height-roiSize)

                while not good_negative_position():
                    # print(lungMask[int(negativeX / 2), int(negativeY / 2), :])
                    negativeX = random.randrange(roiSize, width-roiSize)
                    negativeY = random.randrange(roiSize, height-roiSize)

                negative_save_name = saveName.split('.')[0] + str(i+1) + ".jpg"

                negativeRoi = getRoi(sourceImage, negativeX, negativeY)
                saveImageByName(negative_save_name, negativeRoi, 'Nodules', 'negative')

                highlighted_negative = highlightNodule(sourceImage, negativeX, negativeY)
                saveImageByName(negative_save_name, highlighted_negative, 'Highlights', 'negative')

                highlighted_mask = highlightNodule(lungMask, int(negativeX/2), int(negativeY/2), int(roiSize/2))
                saveImageByName(negative_save_name, highlighted_mask, 'Highlights', 'masks')

            print("Done saving the " + saveName + " images")

    print("Skipped images: ")
    for name in skipped_images:
        print(name)

# Command line arguments in order:
# [round of interest size (int)]
# [operation ('highlight', 'nodule')]
# [Use enhanced images ('yes, no')]
# optional[debug]
# i.e. "python create_test_images.py 256 Nodule"

if __name__ == "__main__":
    main()
    # cv2.waitKey(0)
