import fiftyone as fo

'''

anno_key = "diesel_engine_annotated_parts"

# Step 5: Merge annotations back into FiftyOne dataset

dataset = fo.load_dataset("diesel_engine")
dataset.load_annotations(anno_key)

# Load the view that was annotated in the App
view = dataset.load_annotation_view(anno_key)

session = fo.launch_app(view=view)

session.wait()
'''
json_file = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_dataset/CocoAugJSON.json"
#image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
image_root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_dataset/AugImages"


# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=image_root,
    labels_path=json_file)

session = fo.launch_app(dataset)

session.wait()