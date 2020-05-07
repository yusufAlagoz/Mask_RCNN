import glob, os
import os

from PIL import Image
# Current directory
current_dir_train ="/home/yusuf/Desktop/Datasets/raw_pictures/train_split_folder"
current_dir_val ="/home/yusuf/Desktop/Datasets/raw_pictures/val_split_folder"
current_dir_test ="/home/yusuf/Desktop/Datasets/raw_pictures/test_split_folder"

current_dir = current_dir_train
import json

print(current_dir)
dirs = os.listdir(current_dir)
# Populate train.txt and test.txt
annotation_file = current_dir +'/via_region_data.json'

f = open(os.getcwd()+"/reg.txt", "r")
label = f.readline()

f2 = open(os.getcwd()+"/region2.txt", "r")
label2 = f2.readline()


allpoint_x = [95,1975,1982,91]
allpoint_y = [28,28,1421,1421]

def getFileAttributes():
    print()

def file_size(fname):
    import os
    statinfo = os.stat(fname)
    return statinfo.st_size
data = {}

strList =[]
strList.append("{")
for folderName in dirs:
    tempdir =current_dir+"/"+folderName
    for pathAndFilename in glob.iglob(os.path.join(tempdir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        regions = []
        regions_dict ={}
        regions_dict.update({0:{}})
        regions_dict[0].update({'shape_attributes':{}})
        regions_dict[0].update({'region_attributes': {}})
        regions_dict[0]['shape_attributes'].update({'name':'polygon',"all_points_x": allpoint_x, "all_points_y": allpoint_y})
        regions_dict[0]['region_attributes'].update({'Ship':str(folderName)})

        data[title+ext+str(file_size(pathAndFilename))] = {}
        data[title + ext + str(file_size(pathAndFilename))].update({
            'fileref':"",
            'filename': str(title+ext)  ,
            'size': str(file_size(pathAndFilename)),
            'base64_img_data':"",
            'file_attributes': {},
            'regions': regions_dict,

        })

with open(annotation_file, 'w') as outfile:
    json.dump(data, outfile)

os._exit(0)

