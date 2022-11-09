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
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
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

    def fit(self, val_dataset=None, verbose=True):
        register_coco_instances("diesel_engine_train", {}, "/home/opendr/project-4-at-2022-11-08-12-58-d9d5f8c2/result.json", "/home/opendr/project-4-at-2022-11-08-12-58-d9d5f8c2/images")

        self.cfg.DATASETS.TRAIN = ("diesel_engine_train",)
        
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

    detectron2.fit()



if __name__ == '__main__':
	main()