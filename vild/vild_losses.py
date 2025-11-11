# vild_losses.py(Mark4.2, Mark4.6)

import torch
import torch.nn as nn
import torch.nn.functional as F
from vild_config import AudioViLDConfig

class ViLDLosses:
    """
    ViLD 기반 모델 학습을 위한 커스텀 손실 함수 모듈

    주요 기능:
    - ViLD-text: 텍스트 임베딩 기반 CrossEntropyLoss
    - ViLD-image: 오디오 임베딩 간 거리 기반 L1 + Cosine Distance MSE 혼합 손실
    """
    
    ### 다른 Mark4.x 시리즈와 다른 사항 반영 - Mark4.2와 Mark4.6
    # 1. α 대비치인 text_loss_weight, image_loss_weight 값을 수정.
    # vild_config.py의 AudioViLDConfig 함수에서 값 조정. 
    # 1차: text_loss_weight=0.7, image_loss_weight=0.3 (적용)
    # 2차: text_loss_weight=0.8, image_loss_weight=0.2
    
    # 2. CrossEntropyLoss의 class_weight
    # AudioViLDConfig에 class_weight(예: [w_target, w_others])를 넣어둠.
    # [1.0, 1.5] -> [1.0, 2.0] -> [1.0, 2.5]
    # 가중치를 others 클래스에 두려고 하는 것이라 클래스 인덱스 순서 주의
    # 클래스 인덱스 순서: [target, others] 라는 점 유의.

    def __init__(self, config: AudioViLDConfig):
        self.text_loss_weight = config.text_loss_weight
        self.image_loss_weight = config.image_loss_weight
        
        
        ### Mark4.2와 Mark4.6에만 이러한 변화를 주었다.
        ### config.class_weight 를 넣어 클래스 가중치 추가. 
        # 1) 클래스 가중치 
        self.class_weight = None
        if getattr(config, "class_weight", None) is not None:
            # dtype만 고정, 디바이스는 compute_text_Loss에서 이동.
            self.class_weight = torch.tensor(config.class_weight, dtype=torch.float)
            
        # 2) 기본 CE (weight 없이 ; 필요시 F.cross_entropy로 weight 전달) 
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_text_loss(self, logits, targets): # 텍스트 기반 분류 손실 (CrossEntropyLoss)
        # weight가 있으면 F.cross_entropy로 직접 전달
        if self.class_weight is not None:
            ce = F.cross_entropy(
                logits, targets,
                weight=self.class_weight.to(logits.device)
            )
        else:
            # 평소대로 모듈 사용
            ce = self.ce_loss(logits, targets)
        
        return self.text_loss_weight * ce

    def compute_image_loss(self, student_proj, teacher_embeddings):
        """
        이미지(오디오) 임베딩 간 유사도 손실
        
        Args:
            student_proj (Tensor): [B, D], student 모델 임베딩
            teacher_embeddings (Tensor): [B, D], teacher soft label 임베딩

        Returns:
            Tensor: weighted hybrid distance loss
        """
        l1 = F.l1_loss(student_proj, teacher_embeddings)
        cos_sim = F.cosine_similarity(student_proj, teacher_embeddings, dim=1)
        cos_dist = 1 - cos_sim
        cos_mse = torch.mean(cos_dist ** 2)
        return self.image_loss_weight * (0.5 * l1 + 0.5 * cos_mse)

    def total_loss(self, logits, targets, student_proj, teacher_embeddings):
        """
        전체 손실 계산 (ViLD-text + ViLD-image)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (total, text, image) loss
        """
        text_loss = self.compute_text_loss(logits, targets)
        image_loss = self.compute_image_loss(student_proj, teacher_embeddings)
        total = text_loss + image_loss
        return total, text_loss, image_loss
        