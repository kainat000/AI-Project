import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import cv2
import os, os.path
file_paths=[]
cur_dir=os.getcwd()
for dirpath,dirname,filename in os.walk(cur_dir):
    file_paths.append(dirpath)
trainDataPath=file_paths[3:13]
testDataPath=file_paths[13:24]

#--------------------------------------------------------------------------#

imgs = []
lenOfTrainFile=[]
vImages = [".png"]
count=0
a=0
while count<=9:
    path=trainDataPath[count]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in vImages:
            continue
        imgs.append(cv2.imread(os.path.join(path,f),0))
    b=len(imgs)-a
    newvariable="lengthof",count+1,"iss",b
    lenOfTrainFile.append(b)
    a=len(imgs)
    count+=1

#--------------------------------------------------------------------------------------------------#

xTrain = []
yTrain = []
for img in imgs:
    cropImage = img[(len(img)//2)-30:(len(img)//2)+30]
    img = []
    for row in cropImage:
        img.append(row[(len(row)//2)-30:(len(row)//2)+30])
    img = np.array(img)
    img = img.reshape((1,img.shape[0]*img.shape[1]))
    xTrain.append(img)
xTrain = np.array(xTrain)
xTrain = xTrain.reshape(len(xTrain),-1)
lable=1
for i in lenOfTrainFile:
    for j in range(i):
        yTrain.append(lable)
    lable+=1
yTrain = np.array(yTrain)

#-----------------------------------------------------------------------------------------------#

testimgs = []
lenOfTestFile=[]
vImages = [".png"]
counter=0
a=0
while counter<=9:
    path=testDataPath[counter]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in vImages:
            continue
        testimgs.append(cv2.imread(os.path.join(path,f),0))
    b=len(testimgs)-a
    newvariable="lengthof",counter+1,"iss",b
    lenOfTestFile.append(b)
    a=len(testimgs)
    counter+=1
    
#---------------------------------------------------------------------------------#

xTest = []
yTest = []
for img in testimgs:
    cropImage = img[(len(img)//2)-30:(len(img)//2)+30]
    img = []
    for row in cropImage:
        img.append(row[(len(row)//2)-30:(len(row)//2)+30])
    img = np.array(img)
    img = img.reshape((1,img.shape[0]*img.shape[1]))
    xTest.append(img)
xTest = np.array(xTest)
xTest = xTest.reshape(len(xTest),-1)
lable=1
for i in lenOfTestFile:
    for j in range(i):
        yTest.append(lable)
    lable+=1
yTest = np.array(yTest)

#-----------------------------------------------------------------------------------------------#

classifier = MLPClassifier(hidden_layer_sizes=(500,500,500,500,500,100),activation="relu",learning_rate_init=0.00001,verbose=True,max_iter=500,n_iter_no_change=500)
classifier.fit(xTrain,yTrain)
predict = classifier.predict(xTest)
final = 0
for i in range(len(yTest)):
    final+=1 if predict[i]==yTest[i] else 0
print(final/len(yTest))