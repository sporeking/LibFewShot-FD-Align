import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from ..metric.CoOp import CoOp_head
from core.model.backbone.clip import get_ILF_kmeans_weights_classifier, ImageEncoder, load, get_zeroshot_weight
from datasets.openai_imagenet_temple import openai_imagenet_template
from datasets.class_name import mini_train
import datasets.class_name as class_name
from core.model.backbone.customCLIP import CustomCLIP

class CLIP_context_CoOP(FinetuningModel):
    def __init__(self, 
                 backbone_name = "ViT_B_32",
                 cscale: float = 1.0,
                 cname: str = "openai_imageNet_classnames",
                 **kwargs) -> None:
        super(CLIP_context_CoOP, self).__init__(**kwargs)

        clip_model = ImageEncoder(backbone_name)
        clip_model_, _, _ = load(backbone_name, jit=False)
        classname = getattr(class_name, cname)
        zero_shot_weight = get_zeroshot_weight(clip_model_, openai_imagenet_template, classname)
        self.classifier = CoOp_head(normalize=True, weights=zero_shot_weight)
        self.scale = cscale
        self.clip = CustomCLIP(None,classname,clip_model_)
        
        self.zero_shot_clip = clip_model
        for param in self.zero_shot_clip.parameters():
            param.requires_grad = False
        self.context_classifier = get_ILF_kmeans_weights_classifier(clip_model_, openai_imagenet_template, mini_train, kwargs['cnumber'] if 'cnumber' in kwargs else 1, 60) 
        for param in self.context_classifier.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        del clip_model_
        self.loss_ctx = torch.nn.KLDivLoss()
        #  self.loss_func = F.cross_entropy()

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(image)
        logits = self.classifier(feat)
        return logits, accuracy(logits, target)

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        feat = self.emb_func(image)
        with torch.no_grad():
            zero_feat = self.zero_shot_clip(image)
            
        ctx_loss = self.loss_ctx(torch.log(F.softmax(self.context_classifier(feat), dim=1)), F.softmax(self.context_classifier(zero_feat), dim=1))
        ctx_loss = self.scale * ctx_loss
        
        logits = self.classifier(feat)
        
        loss = F.cross_entropy(logits, target) + ctx_loss
        acc = accuracy(logits, target)
        
        return logits, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        raise NotImplementedError   
        