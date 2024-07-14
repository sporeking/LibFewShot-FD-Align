import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel
# from ..metric.CoOp import CoOp_head
from core.model.backbone.clip import get_ILF_kmeans_weights_classifier, ImageEncoder, load, get_zeroshot_weight
from datasets.openai_imagenet_temple import openai_imagenet_template
from datasets.class_name import mini_train
import datasets.class_name as class_name
from ..backbone.customCLIP import CustomCLIP


class CLIP_context_only_CoOP(FinetuningModel):
    def __init__(self, 
                 backbone_name = "ViT_B_32",
                 cscale: float = 20.0,
                 cname: str = "openai_imageNet_classnames",
                 **kwargs) -> None:
        super(CLIP_context_only_CoOP, self).__init__(**kwargs)

        clip_model = ImageEncoder(backbone_name)
        clip_model_, _, _ = load(backbone_name, jit=False)
        clip_model_ = clip_model_.to(self.device)
        classname = getattr(class_name, cname)
        # zero_shot_weight = get_zeroshot_weight(clip_model_, openai_imagenet_template, classname)
        # self.classifier = CoOp_head(normalize=True, weights=zero_shot_weight)
        self.classifier = CustomCLIP(classname,clip_model_.to(self.device),self.device)
        for name, param in self.classifier.named_parameters():
            param.requires_grad = True

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        del clip_model_

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(image)
        logits = self.classifier.forward(feat)
        return logits, accuracy(logits, target)

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(image)
        logits = self.classifier(feat)
        
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        
        return logits, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        raise NotImplementedError   
        