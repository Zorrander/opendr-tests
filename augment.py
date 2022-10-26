import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import math
import imageio
import glob
from skimage import io
import albumentations as A
import json
import random
from numpy import savez_compressed
import shutil
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


class COCO:
    def __init__(self, json_file, image_root, list_augmentations_file, output_folder, has_keypoints, has_segmentation, has_bbox, augmentation, json_type, max_num_keypoints):
        """
        load dataset
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        """
        self.image_root = image_root
        self.output_folder = output_folder

        if augmentation == True 
            directory = "AugImages" 
            self.augmentation_output = os.path.join(output_folder, directory)
            if not os.path.exists(self.augmentation_output):
                os.mkdir(self.augmentation_output)


        #self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        #self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not imgs_anns == None:
            print('loading annotations into memory...')
            tic = time.time()
            self.dataset = json.load(open(json_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

def augImage(transform,transformCount, ImgId, BboxListArr, SegAndKeypoints, listOfSegCount, catID, maxNumOfKeypoints, listOfkpVisibility, hasKeypoints, hasSeg, hasBBox, validKeypoints, imgPath, outPath, imgName, listOfArea): 
    ## Function used to augment Images and save them as per the COCO Standards
    ## 1. Append class label in BBox List
    ## 2. Transform Image and get list of each annotation style
    ## 3. Albumentation dont process polygon segmentations, so transform them as keypoints
    ## https://albumentations.ai/
    ## 4. Seperate Segmentations from the transformed Keypoints
    ## 5. Save keypoints as per its visibility. TO maintain the count of maximum number of keypoints append some dummy with visibility 0.
    ## https://cocodataset.org/#format-data
    ## 6. Saving the augmented JSON by making a JSON object
    ## 7. Save the augmented image in the output directory
    
    augCOCOjsonAnns = []
    for element in range(0,len(BboxListArr)):
        BboxListArr[element].append(str(catID[element]))
    
    AllKeypoints = []
    image = cv2.imread(imgPath)
    #print(imgPath)
    height, width = cv2.imread(imgPath).shape[:2]
    transformed = transform(image=image, bboxes=BboxListArr, keypoints=SegAndKeypoints)
    transformed_image = transformed['image']
    transformed_keypoints = transformed['keypoints']
    transformed_bboxes = transformed['bboxes']
    transformed_keypoints = np.array(transformed_keypoints)
    
    fileID = 0
    if validKeypoints != 0:
        keyPoints = transformed_keypoints[-validKeypoints:]
        transformed_keypoints = transformed_keypoints[:-validKeypoints]
        AllKeypoints = []
        for ind in range(0,len(keyPoints)):
            AllKeypoints.append(keyPoints[ind][0])
            AllKeypoints.append(keyPoints[ind][1])
            AllKeypoints.append(listOfkpVisibility[ind])

    for indx in range(0,maxNumOfKeypoints - validKeypoints):
        AllKeypoints.append(0)
        AllKeypoints.append(0)
        AllKeypoints.append(0)
    
    keepTrackOfIndex = 0
    AllSegmentations = []
    transformed_keypoints = transformed_keypoints.flatten()
    for ind in range(0,len(listOfSegCount)):
        listOfPoints = transformed_keypoints[keepTrackOfIndex:keepTrackOfIndex+listOfSegCount[ind]]
        keepTrackOfIndex = keepTrackOfIndex + listOfSegCount[ind]
        AllSegmentations.append(listOfPoints.tolist())
    
    ImageID = int(str(ImgId)+str(transformCount))
    
    for eachAnno in range(0,len(catID)):
        if (hasKeypoints == True and hasSeg == True and hasBBox == True) or (hasKeypoints == True and hasSeg == True):
            objAug = {
                "id":str(0),
                "image_id":ImageID,
                "category_id":catID[eachAnno],
                "segmentation": [AllSegmentations[eachAnno]],
                "bbox": list(transformed_bboxes[eachAnno][:4]),
                "ignore":0,
                "iscrowd":0,
                "area": listOfArea[eachAnno],
                "keypoints": AllKeypoints,
                "num_keypoints": validKeypoints
              }
        elif hasKeypoints == False and hasSeg == False and hasBBox == True:
            objAug = {
                "id":str(0),
                "image_id":ImageID,
                "category_id":catID[eachAnno],
                "bbox": list(transformed_bboxes[eachAnno][:4]),
                "ignore":0,
                "iscrowd":0,
                "area": listOfArea[eachAnno],
              }
        elif (hasKeypoints == False and hasSeg == True and hasBBox == True) or (hasKeypoints == False and hasSeg == True and hasBBox == False):
            objAug = {
                "id":str(0),
                "image_id":ImageID,
                "category_id":catID[eachAnno],
                "segmentation": [AllSegmentations[eachAnno]],
                "bbox": list(transformed_bboxes[eachAnno][:4]),
                "ignore":0,
                "iscrowd":0,
                "area": listOfArea[eachAnno],
              }
        augCOCOjsonAnns.append(objAug)
    newImageName = str(ImgId)+str(transformCount)+str(imgName)
    cv2.imwrite(outPath+'/'+newImageName, transformed_image)
    
    return newImageName, augCOCOjsonAnns, ImageID, height, width

def edit_file_name(fname):
    '''
    This function manupilates the label-studio image name in mini JSON
    '''
    head, tail = os.path.split(fname)
    #Use this if you have the same images which you uploaded on LabelStudio.
    tail = tail.split("-")   
    # If you do not have the images, take images from Label studio coco format (In that case do not use the previous line).   
    newFname = tail[1]
    return newFname


def json_min_to_coco_json(imgs_anns,augmentation_output,json_file,image_root,OutputFolder,hasKeypoints,hasBBox,hasSeg,Augmentation,maxNumOfKeypoints,list_augmentations_file):
    '''
    Code to convert the custom JSON-Min structure (Extracted from Label-studio) into the COCO format
    It includes the conversion of polygon segmentation to keypoints as the albumentation library does not support polygon segmentations.
    '''

    if hasKeypoints == True:
        kpString = []
        for i in range(1,maxNumOfKeypoints+1):
            kpWord = "kp"+str(i)
            kpString.append(kpWord)  ## Array containing Keypoint names

    class_names = []  ## List containing Class names
    image_list=[] ## List containing Image defination, As per the COCO standards
    json_label = "polygonlabels" if hasSeg == True else "rectanglelabels"
    
    for img_annotation in imgs_anns:
        fileName = edit_file_name(img_annotation["image"])
        ImgId = img_annotation["id"]
        for CountLabels,valLabels in enumerate(img_annotation["label"]):
            width = valLabels["original_width"]
            height = valLabels["original_height"]
            if len(class_names) == 0:
                class_names.append(valLabels[json_label][0])

            if valLabels[json_label][0] in class_names:
                continue
            else:
                class_names.append(valLabels[json_label][0])
        objImgDesc = {  ## Object containing Image defination
            "width":width,
            "height":height,
            "id":ImgId,
            "file_name":fileName
        }
        image_list.append(objImgDesc)
        listOfCategories = []
        for i in range(0,len(class_names)):
            if hasKeypoints == True and hasSeg == True and hasBBox == True:
                objCat = {  ## Object containing category defination as per the COCO standards
                    "supercategory": "None",
                    "id":i,
                    "name":class_names[i],
                    "keypoints": kpString,
                    "skeleton":[]
                }
            elif hasSeg == True or hasBBox == True:
                objCat = {
                    "supercategory": "None",
                    "id":i,
                    "name":class_names[i]
                }
            else:
                print("Error while category defination, Please specify the format correctly")
            listOfCategories.append(objCat)
            
    listOfAnnotations = [] ## list of all image annotations
    listOfKeyPoints = []  ## list of all keypoints

    ## Code for annotations start here
    
    for Count,val in enumerate(imgs_anns):
        filePath = image_root + str(edit_file_name(val["image"]))
        fileName = str(edit_file_name(val["image"]))
        image = cv2.imread(filePath)
        height, width = cv2.imread(filePath).shape[:2]
        if Augmentation == True:
            listOfAllSeg = [] ## List of all segmentations present in an Image
            listOfAllBbox = [] ## List of all bounding boxes present in an Image
            listOfcatID = [] ## List of all category Ids present in an Image
            listOfSegCount = [] ## List of all the segmentation points count per segmentation in an image
            listOfArea = [] ## List of all the areas a segmentation will cover inside an Image
        getcount = [-1]*len(class_names)  ## Variable to help in developing the catId list
        ImgId = val["id"]
        ValidKeypoint = 0
        totalCountOfValidKeypoint = 0
        if "kp-1" in val:  ## KP-1, as per the mini JSON structure of Label studio.
            totalCountOfValidKeypoint = len(val["kp-1"])  ## Change KP-1 if label studio starts following different naming conventions

        #### code for key points start ###
        if ("kp-1" in val) and (hasKeypoints == True) and totalCountOfValidKeypoint != 0:
            kps = val["kp-1"]
            obkps=[]
            countOfKeypoints = len(kps)
            for _, kp in enumerate(kps):
                if "x" in kp:
                    x_kp = kp["x"] * width/100  ## Conversion to actual parameters from a normalized ones
                    y_kp = kp["y"] * height/100
                    v_kp = 2 #look at keypoints, it is not visible but labeled
                    obkps.append(x_kp)
                    obkps.append(y_kp)
                    obkps.append(v_kp)
                    ValidKeypoint += 1
            if countOfKeypoints < maxNumOfKeypoints:  ## Append rest of the keypoints to maintain the count
                for kp in range(0,(maxNumOfKeypoints-countOfKeypoints)):
                    obkps.append(0)
                    obkps.append(0)
                    obkps.append(0)
        elif hasKeypoints == True and totalCountOfValidKeypoint == 0:
            obkps=[]
            for kp in range(0,maxNumOfKeypoints):
                obkps.append(0)
                obkps.append(0)
                obkps.append(0)
                
                
        #### code for key points ends here ###
        
        ### Code for Segmentation Starts here ###
        
        for CountLabels,valLabels in enumerate(val["label"]):
            bboxArray=[]
            countSegments = 0
            if hasSeg == True:
                for ind in range(0,len(getcount)):
                    if valLabels[json_label][0] == class_names[ind]:
                        getcount[ind] += 1
                        catID = ind
                getPoints = np.array(valLabels["points"]) 
                getPoints[:,0] = getPoints[:,0] * width/100 # Conversion to actual parameters from a normalized ones
                getPoints[:,1] = getPoints[:,1] * height/100
                
                ## Code to check if keypoint exceeds the image height or width and if they are less than 0
                if width in getPoints:
                    #print(getPoints)
                    matchedIndex = np.where(getPoints == width)
                    #print(getPoints,"Sorted")
                    getPoints[matchedIndex[0][0]] = width - 0.001

                if  height in getPoints:
                    matchedIndex = np.where(getPoints == width)
                    getPoints[matchedIndex[0][0]] = height - 0.001

                if (np.any(getPoints <= 0)):
                    matchedIndex = np.where(getPoints <= 0)
                    getPoints[matchedIndex[0][0]] = 0.001
                
            ### Code for Segmentation ends here ###
            
            elif hasBBox == True:
                for ind in range(0,len(getcount)):
                    if valLabels[json_label][0] == class_names[ind]:
                        getcount[ind] += 1
                        catID = ind
                BBoxX = valLabels["x"] * width/100 # Conversion to actual parameters from a normalized ones
                BBoxY = valLabels["y"] * height/100
                BBoxWidth = valLabels["width"] * width/100
                BBoxHeight = valLabels["height"] * height/100
                
            ### Code for Segmentation ends here ###

            ### Code to create image annotation format ###

            if hasSeg == True: 
                poly = (getPoints.flatten()).tolist()
                area = ((np.max(getPoints[:,0])-np.min(getPoints[:,0]))*np.max(getPoints[:,1])-np.min(getPoints[:,1]))
                bboxArray = [np.min(getPoints[:,0]), np.min(getPoints[:,1]), np.max(getPoints[:,0])-np.min(getPoints[:,0]), np.max(getPoints[:,1])-np.min(getPoints[:,1])]
            elif hasBBox == True:
                poly = []
                bboxArray = [BBoxX, BBoxY, BBoxWidth, BBoxHeight]
                area = BBoxWidth*BBoxHeight
            else:
                poly = []
                area = []
                bboxArray = []

            if (hasKeypoints == True and hasSeg == True and hasBBox == True) or (hasKeypoints == True and hasSeg == True):
                obj = {
                    "id":CountLabels,
                    "image_id":ImgId,
                    "category_id":catID,
                    "segmentation": [poly],
                    "bbox": list(bboxArray[:4]),
                    "ignore":0,
                    "iscrowd":0,
                    "area": area,
                    "keypoints": obkps,
                    "num_keypoints": ValidKeypoint
                  }
            elif hasBBox == True and hasKeypoints == False and hasSeg == False:
                obj = {
                    "id":CountLabels,
                    "image_id":ImgId,
                    "category_id":catID,
                    "bbox": list(bboxArray[:4]),
                    "ignore":0,
                    "iscrowd":0,
                    "area": area
                  }
            elif hasBBox == False and hasKeypoints == False and hasSeg == True:
                obj = {
                    "id":CountLabels,
                    "image_id":ImgId,
                    "category_id":catID,
                    "segmentation": [poly],
                    "bbox": list(bboxArray[:4]),
                    "ignore":0,
                    "iscrowd":0,
                    "area": area
                  }
            elif hasSeg == True and hasBBox == True and hasKeypoints == False:
                obj = {
                    "id":CountLabels,
                    "image_id":ImgId,
                    "category_id":catID,
                    "segmentation": [poly],
                    "bbox": list(bboxArray[:4]),
                    "ignore":0,
                    "iscrowd":0,
                    "area": area
                  }
            elif hasKeypoints == True:
                obj = {
                    "id":CountLabels,
                    "image_id":ImgId,
                    "category_id":catID,
                    "ignore":0,
                    "iscrowd":0,
                    "keypoints": obkps,
                    "num_keypoints": ValidKeypoint
                  }

            listOfAnnotations.append(obj)
            if Augmentation == True:
                listOfArea.append(area)
                listOfAllBbox.append(bboxArray)
                for i in range(0, len(poly)-1, 2):
                    listOfAllSeg.append((poly[i], poly[i+1]))
                listOfSegCount.append(len(poly))
                listOfcatID.append(catID)
        if Augmentation == True:  # Image Augmentation 
            imgs_anns, image_list, listOfAllSeg = augment_image(augmentation_output, list_augmentations_file, imgs_anns, image_list,listOfAllSeg, augment_image)

    # Code for the new image starts here
    idCount=0
    for Count,val in enumerate(listOfAnnotations):
        val['id'] = idCount
        idCount+=1
        
    imagesCount=0
    for Count,val in enumerate(image_list):  ## Counting of total number of Images
        imagesCount+=1
    print("Total Number Of Images after Augmentation",imagesCount)
    
    del imgs_anns
    imgs_anns = {  ## Complete COCO Json structure
        "images":image_list,
        "categories":listOfCategories,
        "annotations":listOfAnnotations
    }

    if Augmentation == True:
        directory = "CocoAugJSON"
    else:
        directory = "CocoJSON"
    OutputFolder = OutputFolder
    path_1 = os.path.join(OutputFolder, directory)
    os.mkdir(path_1)

    with open(path_1 + "/"+directory+".json", "w") as outfile:
        json.dump(imgs_anns, outfile)

    print("Coco JSON Saved!")
    return imgs_anns

def makeListOfAnnos(valAnnos):
    ## Extract the annotation from the JSON array
    
    if "segmentation" in valAnnos:
        poly = valAnnos['segmentation'][0]
        for i in range(0, len(poly)-1, 2):
            listOfAllSeg.append((poly[i], poly[i+1]))
        listOfSegCount.append(len(poly))

    if "bbox" in valAnnos:
        listOfAllBbox.append(valAnnos['bbox'])

    if "category_id" in valAnnos:
        listOfcatID.append(valAnnos['category_id'])

    if "area" in valAnnos:
        listOfArea.append(valAnnos['area'])
        
    return listOfSegCount,listOfAllSeg, listOfAllBbox, listOfcatID, listOfArea


def augment_image(augmentation_output, list_augmentations_file, imgs_anns, image_list, listOfAllSeg, hasKeypoints):
    '''
    Make the keypoint List
    Load the augmentations from the file
    Make augmentations
    Save the JSON and the image
    '''
    totalCountOfValidKeypoint = 0
    listOfkpVisibility = []
    
    if hasKeypoints == True:
        obkps = listOfkeypoints
        for i in range(0, len(obkps)-1, 3):
            listOfkpVisibility.append(obkps[i+2])
            if obkps[i+2] != 0:
                totalCountOfValidKeypoint += 1
                listOfAllSeg.append((obkps[i], obkps[i+1]))
   
    augmentation_file = open(list_augmentations_file, 'r')
    augmentation_transformations = augmentation_file.readlines()

    for count, line in enumerate(augmentation_transformations):

        augmentation_style = "A."+str(line.strip())
        transform = A.Compose(
          [eval(augmentation_style)],
          bbox_params=A.BboxParams(format="coco"),
          keypoint_params=A.KeypointParams(format="xy"),
        )
        newImageName, augCOCOJsonAnnos, NewImageID, height, width = augImage(transform,count,ImgId, listOfAllBbox, listOfAllSeg, listOfSegCount, listOfcatID, maxNumOfKeypoints, listOfkpVisibility, hasKeypoints, hasSeg, hasBBox, totalCountOfValidKeypoint,filePath, augmentation_output, fileName, listOfArea)
        objImgDesc = {
            "width":width,
            "height":height,
            "id":NewImageID,
            "file_name":newImageName
        }
        for CountAugs,valAugs in enumerate(augCOCOJsonAnnos):
            image_anns.append(valAugs)
        image_list.append(objImgDesc)
    cv2.imwrite(augmentation_output+'/'+fileName, image)
    return imgs_anns, image_list, listOfAllSeg
    
def innitializeArrays():
    listOfAllSeg
    listOfAllBbox
    listOfcatID
    listOfArea
    listOfSegCount
    listOfkeypoints

#if JSONtype == "COCO":
def augment_coco(imgs_anns,augmentation_output,json_file,image_root,OutputFolder,hasKeypoints,hasBBox,hasSeg,Augmentation,maxNumOfKeypoints):
    ## 1. Extract the lists from previous JSON structure
    ## 2. Convert the polugon segmentations to keypoints
    ## 3. Make Augmentations
    ## 4. Save the COCO Augmented JSON structure and the images
    
    listOffile_name = []
    image_anns = []
    image_def = []
    innitializeArrays()
    
    ImgId = imgs_anns['annotations'][0]["image_id"]
    
    for CountImageName,valImageName in enumerate(imgs_anns['images']):
        if "file_name" in valImageName:
            listOffile_name.append(valImageName['file_name'])
    
    baseImageCount = len(listOffile_name)
    imageCount = 0
    lenOfAnnotations = len(imgs_anns['annotations'])

    for CountAnnos,valAnnos in enumerate(imgs_anns['annotations']):
        if valAnnos['image_id'] == ImgId:
            fileName = listOffile_name[imageCount]
            filePath = image_root + str(fileName)
            image = cv2.imread(filePath)
            listOfkeypoints = valAnnos['keypoints']
            listOfSegCount,listOfAllSeg, listOfAllBbox, listOfcatID, listOfArea = makeListOfAnnos(valAnnos)
            if CountAnnos == lenOfAnnotations-1:
                totalCountOfValidKeypoint = 0
                lastAnno = True
                augment_image()
        else:
            totalCountOfValidKeypoint = 0
            lastAnno = False
            augment_image()
            innitializeArrays()
            ImgId = valAnnos['image_id']
            imageCount+=1
            listOfSegCount,listOfAllSeg, listOfAllBbox, listOfcatID, listOfArea = makeListOfAnnos(valAnnos)

    for CountFinal,valFinal in enumerate(image_anns):
        imgs_anns['annotations'].append(valFinal)
        
    for CountDef,valDef in enumerate(image_def):
        imgs_anns['images'].append(valDef)
    
    idCount=0
    for CountIDCount,valIDCount in enumerate(imgs_anns['annotations']):
        valIDCount['id'] = idCount
        idCount+=1
        valIDCount['bbox'] = valIDCount['bbox'][:4]
    
    imagesCount=0
    for CountImgCount,valImgCount in enumerate(imgs_anns['images']):
        imagesCount+=1
    print("Total Number Of Images after Augmentation",imagesCount)
    
    directory = "CocoAugJSON"
    OutputFolder = OutputFolder
    path_1 = os.path.join(OutputFolder, directory)
    os.mkdir(path_1)

    with open(path_1 + "/"+directory+".json", "w") as outfile:
        json.dump(imgs_anns, outfile)

    print("Coco JSON Saved!")
    
    return imgs_anns

#if split == True:
def split_coco_data(imgs_anns):
    listOffile_name = []
    listOfImageID = []
    for x in imgs_anns:
        listOffile_name.append(x["image"])
        listOfImageID.append(x['id'])
    temp = list(zip(listOfImageID, listOffile_name))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    listOfImageID, listOffile_name = list(res1), list(res2)
    
    listOfImageID_train, listOfImageID_val, listOffile_name_train, listOffile_name_val = train_test_split(listOfImageID, listOffile_name, test_size=0.25, random_state=1)    
    trainSetDef = []
    valSetDef = []
    for x in imgs_anns:
        if x["id"] in listOfImageID_train:
            #print(val)
            trainSetDef.append(x["id"])
        elif x["id"] in listOfImageID_val:
            #print(valSplitDef)
            valSetDef.append(x["id"])

    trainSetAnnos = []
    valSetAnnos = []
    for x in imgs_anns:
        if x["annotation_id"] in listOfImageID_train:
            trainSetAnnos.append(x["annotation_id"])
        elif x["annotation_id"] in listOfImageID_val:
            valSetAnnos.append(x["annotation_id"])
            
    obj_train = {  ## Complete COCO Json structure
        "images":trainSetDef,
        "categories":imgs_anns['categories'],
        "annotations":trainSetAnnos
    }
    
    obj_val = {  ## Complete COCO Json structure
        "images":valSetDef,
        "categories":imgs_anns['categories'],
        "annotations":valSetAnnos
    }
    
    directory = "Train"
    path_Train = os.path.join(OutputFolder, directory)
    os.mkdir(path_Train)
    
    with open(path_Train + "/"+directory+"_COCO.json", "w") as outfile:
        json.dump(obj_train, outfile)
    
    directory = "Images"
    path_Train_Images = os.path.join(path_Train, directory)
    os.mkdir(path_Train_Images)
    
    directory = "Val"
    path_Val = os.path.join(OutputFolder, directory)
    os.mkdir(path_Val)

    with open(path_Val + "/"+directory+"_COCO.json", "w") as outfile:
        json.dump(obj_val, outfile)
        
    directory = "Images"
    path_Val_Images = os.path.join(path_Val, directory)
    os.mkdir(path_Val_Images)
    
    imgsInSource = []
    for (dirname, dirs, files) in os.walk(augmentation_output):
        for filename in files:
            imgsInSource.append(filename)
            
    for imageName in listOffile_name_train:
        if imageName in imgsInSource:
            imageSourcePath = augmentation_output
            shutil.copy(os.path.join(imageSourcePath, imageName), os.path.join(path_Train_Images, imageName))
            os.remove(imageSourcePath+"/"+imageName)
    for imageName in listOffile_name_val:
        if imageName in imgsInSource:
            imageSourcePath = augmentation_output
            shutil.copy(os.path.join(imageSourcePath, imageName), os.path.join(path_Val_Images, imageName))
            os.remove(imageSourcePath+"/"+imageName)
    #shutil.rmtree(augmentation_output, ignore_errors=True)
    print("Train and Val Dataset saved!")

def main():
    ## Ask the below inputs from the User
    json_file = input("json_file: ") # "../Dataset/miniJSON/AnnotatedJSON.json"

    image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
    list_augmentations_file = input("list_augmentations_file: ") 
    output_folder = input("output_folder: ") # "../Dataset"
    hasKeypoints_input = input("hasKeypoints: ") # True
    if hasKeypoints_input == "True":
        hasKeypoints = True 
    hasBBox_input = input("hasBBox: ") # True
    if hasBBox_input == "True":
        hasBBox = True
    hasSeg_input= input("hasSeg: ") # True
    if hasSeg_input == "True":
        hasSeg = True
    Augmentation_input = input("Augmentation: ") # True
    if Augmentation_input == "True":
        Augmentation = True
    JSONtype = input("JSONtype: ") # "Custom" ## Default COCO or Custom
    if JSONtype == "COCO" and not Augmentation:
        print("Since the JSON type is COCO, please select Augmentations or train the COCO dataset directly")
        return
    maxNumOfKeypoints = input("maxNumOfKeypoints: ") # 25  ## Ask only if keypoints is true
    maxNumOfKeypoints = int(maxNumOfKeypoints)

    coco = Coco(json_file, image_root, list_augmentations_file, output_folder, has_keypoints, has_segmentation, has_bbox, augmentation, json_type, max_num_keypoints)

    if JSONtype == "Custom":
        imgs_anns = coco.json_min_to_coco_json()
    elif JSONtype == "COCO":
        imgs_anns = coco.augment_coco()
    
    coco=COCO(imgs_anns)
    image_ids = coco.getImgIds(catIds=[2])
    print("image_ids ", image_ids)

    # split_coco_data(imgs_anns)

    '''
    if Augmentation == True:
        JsonPathAug = output_folder +"CocoAugJSON"
        'shutil'.rmtree(augmentation_output, ignore_errors=True)
        shutil.rmtree(JsonPathAug, ignore_errors=True)
    '''

if __name__ == '__main__':
    main()