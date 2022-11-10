# fiftyone imports 
import fiftyone as fo
import fiftyone.utils.random as four

# Detectron imports
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL

import fiftyone.utils.random as four

class Detectron2Learner(Learner):
    supported_backbones = ["resnet"]

    def __init__(self, lr=0.00025, batch_size=200, img_per_step=2, weight_decay=0.00008,
                       momentum=0.98, gamma=0.0005, norm="GN", num_workers=2, num_keypoints=25, 
                       iters=4000, threshold=0.9, loss_weight=1.0, device='cuda', temp_path="temp", backbone='resnet'):
        super(Detectron2Learner, self).__init__(lr=lr, threshold=threshold, 
                                                batch_size=batch_size, device=device, 
                                                iters=iters, temp_path=temp_path, 
                                                backbone=backbone)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
        self.cfg.MODEL.MASK_ON = True
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_step
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.WEIGHT_DECAY = weight_decay
        self.cfg.SOLVER.GAMMA = gamma
        self.cfg.SOLVER.MOMENTUM = momentum
        self.cfg.SOLVER.MAX_ITER = iters
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size   # faster, and good enough for this toy dataset
        self.cfg.MODEL.SEM_SEG_HEAD.NORM = norm
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = loss_weight


    def __export_split(self, dataset, tag):
        root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/"
        extension = ".json"
        samples = dataset.match_tags(tag)
        samples.export(
            dataset_type=fo.types.COCODetectionDataset,
            labels_path= f"{root}{tag}{extension}",
            abs_paths=True,
        )

    def __prepare_dataset(self, dataset):
        four.random_split(dataset, {"train": 0.7, "test": 0.2, "val": 0.1})
        self.__export_split(dataset, "train")
        self.__export_split(dataset, "test")
        self.__export_split(dataset, "val")
        register_coco_instances("diesel_engine_train", {}, "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/train.json", "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/AugImages")
        register_coco_instances("diesel_engine_val", {}, "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/val.json", "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/AugImages")
        self.cfg.DATASETS.TRAIN = ("diesel_engine_train",)
        self.cfg.DATASETS.TEST = ("diesel_engine_val")

    def fit(self, dataset, verbose=True):
        self.__prepare_dataset(dataset)

        trainer = DefaultTrainer(self.cfg)

        trainer.resume_or_load(resume=False)

        trainer.train()

    def infer(self, img_data):
        pass

    def load(self, verbose=True):
       pass


    def save(self, path, verbose=False):
        pass

    def download(self, path=None, mode="pretrained", verbose=False, 
                        url=OPENDR_SERVER_URL + "/perception/object_detection_2d/detectron2/"):
        pass

    def eval(self, json_file, image_root):
        pass

    def optimize(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()

    def reset(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()


def main():
    detectron2 = Detectron2Learner()
    json_file = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/CocoAugJSON.json"
    #image_root = input("image_root: ") # "../Dataset/AnnotatedImages/"
    image_root = "/home/opendr/project-7-at-2022-11-09-14-23-12119809/new_new_dataset/AugImages"
    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_root,
        labels_path=json_file)
    detectron2.fit(dataset)



if __name__ == '__main__':
	main()