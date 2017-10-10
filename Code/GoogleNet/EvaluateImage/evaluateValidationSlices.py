import evaluate_image_slice
import EvaluateWholeImage
import glob
import os
import cv2
import shutil
import openpyxl


class data(object):
    name = ""
    positive = 0
    negative = 0


# constants
positiveCertainty = 0.6
negativeCertainty = 0.4

correctlyClassifiedImagesPath = os.path.join(os.path.dirname(__file__), os.pardir, "Results", "Correct")
incorrectlyClassifiedImagesPath = os.path.join(os.path.dirname(__file__), os.pardir, "Results", "Incorrect")
undecidedClassifiedImagesPath = os.path.join(os.path.dirname(__file__), os.pardir, "Results", "Undecided")


def deleteOldFiles():
    def deleteFiles(deleteFiles):
        for f in deleteFiles:
            # print(f)
            os.remove(f)

    deleteFiles(glob.glob(correctlyClassifiedImagesPath + "\\*"))
    deleteFiles(glob.glob(incorrectlyClassifiedImagesPath + "\\*"))
    deleteFiles(glob.glob(undecidedClassifiedImagesPath + "\\*"))


def getPathFromFileName(name):
    image_paths = glob.glob("d:\\School\\2017Onlab\\ProjectLab\\Code\\Creating_Training_Images\\ValidateImages\\ImageSlices\\*")
    for path in image_paths:
        if name in path:
            return path
    print("Name" + name + "not found in possible paths")


def calculateResults(healthy, tumors, cantTell):
    result = EvaluateWholeImage.resultObj()

    for data in healthy:
        if 'Positive' in data.name:
            result.false_neg.append(data.name)
            shutil.copy(getPathFromFileName(data.name), incorrectlyClassifiedImagesPath + "\\" + data.name + str(data.positive) + ".png")
        if 'Negative' in data.name:
            result.true_neg.append(data.name)
            shutil.copy(getPathFromFileName(data.name), correctlyClassifiedImagesPath + "\\" + data.name + str(data.positive) + ".png")

    for data in tumors:
        if 'Positive' in data.name:
            result.true_pos.append(data.name)
            shutil.copy(getPathFromFileName(data.name), correctlyClassifiedImagesPath + "\\" + data.name + str(data.positive) + ".png")
        if 'Negative' in data.name:
            result.false_pos.append(data.name)
            shutil.copy(getPathFromFileName(data.name), incorrectlyClassifiedImagesPath + "\\" + data.name + str(data.positive) + ".png")
    
    for data in cantTell:
        result.undecided.append(data.name)
        shutil.copy(getPathFromFileName(data.name), undecidedClassifiedImagesPath + "\\" + data.name + str(data.positive) + ".png")

    result.calculateIndicators()
    result.saveToFile()


def saveFeatureMaps(positive, negative):
    wb = openpyxl.load_workbook('d:\\School\\2017Onlab\\ProjectLab\\Code\\GoogleNet\\EvaluateImage\\' + "FeatureMaps.xlsx")
    ws = wb.active
    for fm in positive:
        tmpList = fm.tolist()
        tmpList.insert(0, 'positive')
        ws.append(tmpList)
    for fm in negative:
        tmpList = fm.tolist()
        tmpList.insert(0, 'negative')
        ws.append(tmpList)

    

    wb.save('d:\\School\\2017Onlab\\ProjectLab\\Code\\GoogleNet\\EvaluateImage\\' + "FeatureMaps.xlsx")


def Main():
    deleteOldFiles()

    evaluate_image_slice.init_tf("D:\\tmp\\output_graph.pb", ["positive", "negative"])

    image_paths = glob.glob("d:\\School\\2017Onlab\\ProjectLab\\Code\\Creating_Training_Images\\ValidateImages\\ImageSlices\\*")
    tumors = []
    healthy = []
    cantTell = []
    featureMapsPositive = []
    featureMapsNegative = []

    for path in image_paths:
        image_name = os.path.basename(path)
        image = cv2.imread(path)
        prediction = evaluate_image_slice.evaluateSlice(image)
        
        featureMap = evaluate_image_slice.featureMapOfSlice(image)
        if 'Positive' in image_name:
            featureMapsPositive.append(featureMap[0])
        elif 'Negative' in image_name:
            featureMapsNegative.append(featureMap[0])

        resultData = data()
        resultData.name = image_name
        resultData.positive = prediction[0][0]
        resultData.negative = prediction[0][1]
        if prediction[0][0] >= positiveCertainty:
            tumors.append(resultData)
        elif prediction[0][0] <= negativeCertainty:
            healthy.append(resultData)
        else:
            cantTell.append(resultData)
    calculateResults(healthy=healthy, tumors=tumors, cantTell=cantTell)
    saveFeatureMaps(featureMapsPositive, featureMapsNegative)
if __name__ == "__main__":
    Main()