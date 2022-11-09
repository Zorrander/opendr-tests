import os
import cv2
import json
import numpy as np
import fiftyone as fo
import albumentations as A


class COCO:
    '''
            COCO Format :
         {
            "images": [
                {
                    "file_name": "",
                    "height": ,
                    "width": ,
                    "id": 
                },
                ...
            "annotations": [
                {
                    "area": ,
                    "iscrowd": ,
                    "image_id": ,
                    "bbox": [
                        100,
                        100,
                        100,
                        100
                    ],
                    "category_id": 0,
                    "id": 1,
                    "ignore": 0,
                    "segmentation": []
                }
                ...
            "categories": [
                {
                    "supercategory": "",
                    "id": ,
                    "name": ""
                }
            }
    '''
    def __init__(self, json_file, image_root, augmentations_file, output_folder):
        """
        load dataset
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        """
        self.augmentations_file = augmentations_file
        self.output_folder = output_folder
        self.json_file = json.load(open(json_file, 'r'))
        directory = "AugImages" 
        augmentation_output = os.path.join(output_folder, directory)
        if not os.path.exists(augmentation_output):
            os.makedirs(augmentation_output)
        self.augmentation_output = augmentation_output

        # Import the dataset
        self.dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=image_root,
            labels_path=json_file,
            name="coco_diesel_engine",
        )

        self.image_id_start = self.json_file['images'][-1]['id']
        self.annotation_id_start = self.json_file['annotations'][-1]['id']

    def augment_dataset(self):
        self.dataset.compute_metadata()
        if not self.augmentations_file == None:   
            print('loading augmentation steps...') 
            augmentations = open(self.augmentations_file, 'r')
            augmentation_transformations = augmentations.readlines()
        self.generate_new_images(augmentation_transformations)

    def generate_new_images(self, augmentation_transformations):
        cnt = 0
        for sample in self.dataset:
            print("Image {}/{} - {}".format(cnt, len(self.dataset), self.json_file['images'][cnt]['file_name']), end=" ", flush=True)
            img = cv2.imread(sample.filepath, 1)
            filename = os.path.join(self.augmentation_output, self.json_file['images'][cnt]['file_name'] + ".jpeg")
            cv2.imwrite(filename, img)
            os.rename(filename, os.path.splitext(filename)[0])
            for count, line in enumerate(augmentation_transformations):
                print("=", end="", flush=True)
                self.apply_transformation(img, sample, count, line)
            cnt+=1            
            print()

        with open(os.path.join(self.output_folder, "CocoAugJSON.json"), 'w') as outfile:
            json.dump(self.json_file, outfile)

    def apply_transformation(self, img, sample, transform_id, transform):
        augmentation_style = "A."+str(transform.strip())
        transformation = A.Compose(
          [eval(augmentation_style)],
          bbox_params=A.BboxParams(format="coco", label_fields=[]),
          keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        self.transform(img, sample, transformation, transform_id)

    def transform(self, img, sample, albu_transform, transformCount):         
        height = sample.metadata["height"]
        width = sample.metadata["width"]

        bboxes, img_keypoints, category_ids = [], [], []
        keypoint_record = []

        for det in sample.segmentations.detections:
            tlx, tly, w, h = det.bounding_box
            bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
            bboxes.append(bbox)
            polyline = det.to_polyline()
            det_keypoints = [(x*width, y*height) for x, y in polyline.points[0]]
            img_keypoints.extend(det_keypoints)
            category_ids.append(det.label)

        print(img_keypoints)

        transformed = albu_transform(image=img, bboxes=bboxes, keypoints=img_keypoints, category_ids=category_ids)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_categories = transformed['category_ids']
        transformed_keypoints = transformed['keypoints']
        print("transformed keypoints ", transformed_keypoints)

        new_name = f"{sample.id}{str(transformCount)}"
        filename = os.path.join(self.augmentation_output, new_name + ".jpeg")
        cv2.imwrite(filename, transformed_image)
        os.rename(filename, os.path.splitext(filename)[0])
        self.image_id_start += 1

        self.json_file['images'].append( {
            "width":width,
            "height":height,
            "id":self.image_id_start,
            "file_name":new_name
        })    

        for bbox, poly, cat_it in zip(transformed_bboxes,transformed_keypoints, transformed_categories):
            self.annotation_id_start += 1
            cat_id=0 if cat_it == "bolt" else 1
            self.json_file['annotations'].append({
                "id":self.annotation_id_start,
                "image_id":self.image_id_start,
                "category_id": cat_id,
                "segmentation": poly,
                "bbox": bbox,
                "ignore":0,
                "iscrowd":0,
                "area": int(bbox[-1]*bbox[-2]),
            })  


def main():
    ## Ask the below inputs from the User
    #json_file = input("json_file: ") # "../Dataset/miniJSON/AnnotatedJSON.json"
    json_file = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/result.json"
    #image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
    image_root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/images/"
    #list_augmentations_file = input("list_augmentations_file: ") 
    list_augmentations_file = "/home/opendr/Gaurang/engineAssembly/new_dataset/Augmentations"
    #output_folder = input("output_folder: ") # "../Dataset"
    output_folder = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_dataset/"

    coco = COCO(json_file, image_root, list_augmentations_file, output_folder)

    coco.augment_dataset()


if __name__ == '__main__':
    main()