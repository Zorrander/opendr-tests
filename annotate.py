import fiftyone as fo

name = "diesel_engine"
anno_key = "diesel_engine_annotated_parts"

try: 
    dataset = fo.load_dataset(name)
    dataset.load_annotations(anno_key)

except:
    # The directory containing the source images
    dataset_dir = input("/path/to/dataset: ") 

    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.ImageDirectory,
        name=name,
    )
    dataset.persistent = True
    
finally:
    label_field = "tags",
    label_type = "polygons"
    classes = ["bolt", "rocker_arm"]

    view = dataset.view()

    view.annotate(
        anno_key,
        backend="labelstudio",
        url="http://localhost:8080",
        label_field=label_field,
        label_type=label_type,
        classes=classes,
        launch_editor=True,
    )

