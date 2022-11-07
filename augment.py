import os
import cv2
import json
import albumentations as a


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
        self.image_root = image_root
        self.output_folder = output_folder
        self.augmentations_file = augmentations_file
  
        directory = "AugImages" 
        augmentation_output = os.path.join(output_folder, directory)
        if not os.path.exists(augmentation_output):
            os.makedirs(augmentation_output)
        self.augmentation_output = augmentation_output

        self.dataset,self.anns,self.imgs = dict(),dict(),dict()
        self.imgToAnns = defaultdict(list)

        if not json_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(json_file, 'r'))
            # assert type(self.dataset)==dict, 'annotation file format {} not supported'.format(type(self.dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            #Get Last Annotation id
            self.annotation_id_start = self.dataset['annotations'][-1]['id']
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, imgs = {}, {}
        imgToAnns = defaultdict(list)

        for img in self.dataset:
            imgs[img['id']] = img['image']
            anns[img['annotation_id']] = img['label']
            imgToAnns[img['id']] = img['annotation_id']

        print('index created!')
        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs

    def __get_img(self, img_id):
        '''
        :param imgIds (int array) : get imgs for given id
        :return: 
        '''
        json_path = self.imgs[img_id]
        head, tail = os.path.split(json_path)
        #Use this if you have the same images which you uploaded on LabelStudio.
        tail = tail.split("-")   
        # If you do not have the images, take images from Label studio coco format (In that case do not use the previous line).   
        filename = tail[1]
        return filename

    def __get_ann(self, ann_id):
        '''
        :param imgIds (int array) : get imgs for given id
        :return: 
        '''
        return self.anns[ann_id]

    def __generate_image_name(self, img_name, transform_id):
        img_split = img_name.split(".")
        img_num = img_split[0]
        img_extension = "."+img_split[1]
        new_name = f"{img_num}{str(transform_id)}{img_extension}" 
        new_id = int(f"{img_num}{str(transform_id)}")
        return new_id, new_name

    def augment_dataset(self):
        if not self.augmentations_file == None:   
            print('loading augmentation steps...') 
            augmentations = open(self.augmentations_file, 'r')
            augmentation_transformations = augmentations.readlines()
        self.generate_new_images(augmentation_transformations)

    def generate_new_images(self, augmentation_transformations):
        for cnt, img_id in enumerate(self.imgs):
            print("Image {}/{}".format(cnt+1, len(self.imgs)), end=" ", flush=True)
            img = cv2.imread(os.path.join(self.image_root, self.__get_img(img_id)), 1)
            cv2.imwrite(self.augmentation_output, img)
            for count, line in enumerate(augmentation_transformations):
                print("=", end="", flush=True)
                new_id, new_name = self.apply_transformation(img_id, count, line)            
            print()
        with open(os.path.join(self.output_folder, "CocoAugJSON.json"), 'w') as outfile:
            json.dump(self.dataset, outfile)

    def apply_transformation(self, img_id, transform_id, transform):
        augmentation_style = "A."+str(transform.strip())
        transformation = A.Compose(
          [eval(augmentation_style)],
          bbox_params=A.BboxParams(format="coco"),
          keypoint_params=A.KeypointParams(format="xy"),
        )
        self.transform(img_id, transformation, transform_id)

    def transform(self, img_id, albu_transform, transformCount): 
        transformed = albu_transform(image=image, bboxes=[], keypoints=[])
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_categories = transformed['category_ids']
        #transformed_keypoints = np.array(transformed['keypoints'])

        height, width = transformed_image.shape[:2]
        new_id, new_name = self.__generate_image_name(self.__get_img(img_id), transformCount)

        cv2.imwrite(os.path.join(self.augmentation_output, new_name), transformed_image)

        self.dataset['images'].append( {
            "width":width,
            "height":height,
            "id":new_id,
            "file_name":new_name
        })    
        
        self.annotation_id_start += 1
        for bbox,cat_it in zip(transformed_bboxes,transformed_categories):
            self.dataset['annotations'].append({
                "id":self.annotation_id_start,
                "image_id":new_id,
                "category_id": int(cat_it),
                "segmentation": [],
                "bbox": bbox,
                "ignore":0,
                "iscrowd":0,
                "area": int(bbox[-1]*bbox[-2]),
            })  


def main():
    ## Ask the below inputs from the User
    #json_file = input("json_file: ") # "../Dataset/miniJSON/AnnotatedJSON.json"
    json_file = "/home/opendr/Gaurang/engineAssembly/new_dataset/miniJSON/AnnotatedJSON.json"
    #image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
    image_root = "/home/opendr/Gaurang/engineAssembly/new_dataset/AnnotatedImages/"
    #list_augmentations_file = input("list_augmentations_file: ") 
    list_augmentations_file = "/home/opendr/Gaurang/engineAssembly/new_dataset/Augmentations"
    #output_folder = input("output_folder: ") # "../Dataset"
    output_folder = "/home/opendr/alex/engineAssembly/new_dataset/"

    coco = COCO(json_file, image_root, list_augmentations_file, output_folder)

    coco.augment_dataset()


if __name__ == '__main__':
    main()