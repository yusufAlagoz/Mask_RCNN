# import cv2
import os,glob
import os
import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sn


from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
cwd = os.getcwd()
print(cwd)
print(ROOT_DIR)


modelFile = "mask_rcnn_ship_0499.h5"
current_dir = "./test"
modelfile_list = modelFile.split("_")
dirs = os.listdir(current_dir)
mainFolder ="./Test_Sonuçları/"
# os.mkdir(mainFolder)
mapScoresFile = "mapScores.csv"
confMatrixFile = "confMatrixCounts"
confMatrixNormalizedFile= "confMatrixNormalized"
confMatrixImageFileNorm = mainFolder +"confMatrixImageNorm.png"
confMatrixImageFile = mainFolder+ "confMatrixImage.png"




import ShipClassification
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = ShipClassification.ShipClassificationConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE =0.3
    NUM_CLASSES = 1 + 12
    BACKBONE ="resnet101"
config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# Create model in inference mode
print(MODEL_DIR)
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

import numpy as np
import skimage.draw
classNames = {}
classNames[1] = "Balıkçı"
classNames[2] = "Askeri Tip-1"
classNames[3] = "Askeri Tip-2"
classNames[4] = "Askeri Tip-3"
classNames[5] = "Tanker"
classNames[6] = "Destek_Gemisi"
classNames[7] = "Askeri Tip-4"
classNames[8] = "Askeri Tip-5"
classNames[9] = "Askeri Tip-6"
classNames[10] = "Askeri Tip-7"
classNames[11] = "Konteyner"
classNames[12] = "Askeri Tip-8"


def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if (class_id == 1):
            colors.append((.904, .204, .204))
        elif (class_id == 2):
            colors.append((.164, .196, .0))
        elif (class_id == 3):
            colors.append((.250, .104, .0))
        elif (class_id == 4):
            colors.append((.122, .59, .63))
        elif (class_id == 5):
            colors.append((.229, .20, .0))
        elif (class_id == 6):
            colors.append((.27, .61, .226))
        elif (class_id == 7):
            colors.append((.0, .0, .0))
        elif (class_id == 8):
            colors.append((.130, .90, .44))
        elif (class_id == 9):
            colors.append((.229, .20, .0))
        elif (class_id == 10):
            colors.append((.0, .71, .169))
        elif (class_id == 11):
            colors.append((.96, .169, .23))
        elif (class_id == 12):
            colors.append((.0, .71, .169))

    return colors

WEIGHTS_PATH = modelFile  # TODO: update this path
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
with tf.device(DEVICE):
    model.load_weights(weights_path, by_name=True)


listOfValues=[]



"""
Klasörler ve isimler yazılıyor.
"""
labels = {}
classes =[]
f = open("labels.txt", "r")
index=0
for x in f:
    labels.update({x.replace("\n",""):index})
    index=index+1
    classes.append(x.replace("\n",""))


def plot_confusion_matrix(frmt,title,filename,cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10),dpi=500)
    sn.set(font_scale=1)
    cm_df_decisionRatesWOConfMetrics = pd.DataFrame(cm, index=classes, columns=classes)
    p = sn.heatmap(cm_df_decisionRatesWOConfMetrics, annot=True, cmap='Blues', annot_kws={"size":10}, linewidths=0.01, cbar=False, fmt=frmt)
    p.set_xticklabels(p.get_xticklabels(),rotation=45,va="top", ha="right")
    # plt.text(12, -0.8, title, fontsize=10, color='Black', fontstyle='italic', va="bottom", ha="center")
    plt.tight_layout()
    plt.savefig(filename)
    return plt


conf_matrix = np.zeros((12,12))



def add_To_Conf_Matrix(result,classNo):
    if (len(result)>0):
        valueOfPredict = r['class_ids'][0]
        conf_matrix[classNo-1][valueOfPredict-1] = conf_matrix[classNo-1][valueOfPredict-1] +1
        returnVal= True
    else:
        returnVal= False
    return returnVal

totalFile= 0
totalLoss =0
for folderName in range (1,13):
    tempdir =current_dir+"/"+str(folderName)
    dir_len = len([f for f in glob.glob(tempdir + "**/*.jpg", recursive=True)])

    for pathAndFilename in glob.iglob(os.path.join(tempdir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        image = skimage.io.imread(pathAndFilename)

        results = model.detect([image])  # , verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    classNames, r['scores'],
                                    colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
        print("Sınıflar: ", r['class_ids'], " Scores:", r['scores'])
        # plt.show()
        plt.savefig(current_dir+"/Sonuç_resimler/"+str(folderName)+"/"+title+".png")

        # print(result)
        returnValue = add_To_Conf_Matrix(r['class_ids'],folderName)
        if (returnValue==True):
            totalFile=totalFile +1
        else:
            totalLoss = totalLoss +1

#Confusion matrix normalize ediliyor.
row_sums = conf_matrix.sum(axis=1)
conf_matrix_normed = (conf_matrix / row_sums[:, np.newaxis]) *100

#Accuracy Computing

sumsTotal = conf_matrix.sum(axis=0)
for i in range(0,12):
    temp_dict={}
    temp_TP = (conf_matrix[i][i])
    temp_FP = (sumsTotal[i]-temp_TP)
    temp_Precision = temp_TP/(temp_FP+temp_TP)
    # temp_dict.update({"ClassLabel":classes[i]})
    temp_dict.update({'TP':temp_TP})
    temp_dict.update({'FP':temp_FP})
    temp_dict.update({'mAP':temp_Precision})
    listOfValues.append(temp_dict)

sumOfPR = 0
index = 0;

np.save(mainFolder+confMatrixFile,conf_matrix,True,True)
np.save(mainFolder+confMatrixNormalizedFile,conf_matrix_normed,True,True)
########################### Avg mAp hesaplanıp yazılıyor###############3333333
import math
for i in range(0, len(listOfValues)):
    if (math.isnan(listOfValues[i]['mAP'])):
        continue
    else:
        sumOfPR = sumOfPR + listOfValues[i]['mAP']
        index= index +1
dictAvgMap={}
mAP = sumOfPR /index
dictAvgMap.update({"avgMap":mAP})
listOfValues.append(dictAvgMap)

############################ değerler dosyaya yazılıyor  #############3
import pandas as pd
df = pd.DataFrame(listOfValues)
print(df)
df.to_csv(mainFolder+mapScoresFile)



##################3 confusion matrix çizdiriliyor ve kaydediliyor ######################
np.save(mainFolder+confMatrixFile,conf_matrix,True,True)
np.save(mainFolder+confMatrixNormalizedFile,conf_matrix_normed,True,True)
#
plot_confusion_matrix(frmt=".0f",cm=conf_matrix,classes=classes,filename=confMatrixImageFile,title="Total File: "+str(totalFile))
plot_confusion_matrix(frmt='.2f',cm=conf_matrix_normed,classes=classes,filename=confMatrixImageFileNorm,title="Total File: "+str(totalFile) )

print("Prediction Finished...")
os._exit(0)





