
@echo off
call %~dp0/../Creating_Training_Images/create_test_images.bat

call %~dp0/RetrainInception/retrain_inception.bat %1

python %~dp0/EvaluateImage/EvaluateWholeImage.py d:\School\2017Onlab1\Code\Creating_Training_Images\EveryImage\JPCLN009Result.png

REM robocopy D:\tmp d:\School\2017Onlab1\Code\GoogleNet\Model\ /S /E /MOVE
