import torch
import copy

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from evaluator import VOCDetectionEvaluator



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return VOCDetectionEvaluator(dataset_name)

    # @staticmethod
    # def mapper(dataset_dict):
    #     # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    #     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    #     image = utils.read_image(dataset_dict["file_name"], format="BGR")
    #     image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    #     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    #
    #     annos = [
    #         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    #         for obj in dataset_dict.pop("annotations")
    #         if obj.get("iscrowd", 0) == 0
    #     ]
    #     instances = utils.annotations_to_instances(annos, image.shape[:2])
    #     dataset_dict["instances"] = utils.filter_empty_instances(instances)
    #     return dataset_dict
    #
    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=Trainer.mapper)

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     """
    #     Returns:
    #         iterable
    #
    #     It now calls :func:`detectron2.data.build_detection_test_loader`.
    #     Overwrite it if you'd like a different data loader.
    #     """
    #     return build_detection_test_loader(cfg, dataset_name)
