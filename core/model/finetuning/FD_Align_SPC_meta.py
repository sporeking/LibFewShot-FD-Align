import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from ..metric.proto_net import ProtoLayer
from core.model.backbone.clip import get_ILF_kmeans_weights_classifier, ImageEncoder, load
from datasets.openai_imagenet_temple import openai_imagenet_template
from datasets.class_name import mini_train

class CLIP_context(FinetuningModel):
    def __init__(self, 
                 backbone_name = "ViT_B_32",
                 cscale: float = 1.0,
                 **kwargs) -> None:
        super(CLIP_context, self).__init__(**kwargs)

        self.classifier = ProtoLayer()
        clip_model = ImageEncoder(backbone_name)
        clip_model_, _, _ = load(backbone_name, jit=False)
        self.scale = cscale
        
        self.zero_shot_clip = clip_model
        for param in self.zero_shot_clip.parameters():
            param.requires_grad = False
        self.context_classifier = get_ILF_kmeans_weights_classifier(clip_model_, openai_imagenet_template, mini_train, kwargs['cnumber'] if 'cnumber' in kwargs else 1, 60) 
        for param in self.context_classifier.parameters():
            param.requires_grad = False
        del clip_model_
        self.loss_ctx = torch.nn.KLDivLoss()
        self.loss_func = F.cross_entropy()

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
            
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        
        output = self.classifier(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
            mode="cos_sim",
        ).reshape(-1, self.way_num)
        
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc

        # return output, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(image)
        with torch.no_grad():
            zero_feat = self.zero_shot_clip(image)
            
        feat = F.normalize(feat, dim=1)
        zero_feat = F.normalize(zero_feat, dim=1)
        ctx_loss = self.loss_ctx(torch.log(F.softmax(self.context_classifier(feat), dim=1)), F.softmax(self.context_classifier(zero_feat), dim=1))
        ctx_loss = self.scale * ctx_loss
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        logits = self.classifier(query_feat, support_feat, self.way_num, self.shot_num, self.query_num, mode="cos_sim")
        loss = self.loss_func(logits, query_target)
        loss = loss + ctx_loss
        acc = accuracy(logits, query_target)
        
        return logits, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        raise NotImplementedError   
        