"""
선호도 모델 (Preference Model)

사용자 선호도를 학습하고 모델링하는 시스템입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time

from .preference_learner import PreferenceLearner
from .preference_memory import PreferenceMemory, PreferenceData

logger = logging.getLogger(__name__)


class PreferenceEncoder(nn.Module):
    """선호도 인코더"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 입력 인코더
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 최종 인코딩 레이어
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """선호도 데이터 인코딩"""
        input_encoded = self.input_encoder(input_data)
        final_encoded = self.final_encoder(input_encoded)
        return final_encoded


class PreferencePredictor(nn.Module):
    """선호도 예측기"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 선호도 예측 네트워크
        self.preference_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 신뢰도 예측기
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, encoded_preference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """선호도 예측"""
        preference_prediction = self.preference_predictor(encoded_preference)
        confidence = self.confidence_predictor(encoded_preference)
        return preference_prediction, confidence


class PreferenceModel(nn.Module):
    """선호도 모델 메인 클래스"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # 모델 파라미터
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.embedding_dim = config.get('embedding_dim', 64)
        
        # 구성 요소 초기화
        self.encoder = PreferenceEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        )
        
        self.predictor = PreferencePredictor(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim // 2
        )
        
        # 고급 구성 요소
        self.learner = PreferenceLearner(
            embedding_dim=self.embedding_dim,
            learning_rate=config.get('learning_rate', 1e-4),
            memory_size=config.get('learner_memory_size', 1000)
        )
        
        self.memory = PreferenceMemory(
            max_size=config.get('memory_size', 10000),
            memory_type=config.get('memory_type', 'fifo'),
            similarity_threshold=config.get('similarity_threshold', 0.8)
        )
        
        # 모델 상태
        self.training_mode = True
        self.update_count = 0
        
        logger.info("고급 선호도 모델이 초기화되었습니다.")
        
    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """선호도 예측"""
        encoded = self.encoder(input_data)
        prediction, confidence = self.predictor(encoded)
        return prediction, confidence
        
    def update(self, outputs: Dict[str, Any], preferences: Optional[Dict[str, Any]] = None, rewards: Optional[torch.Tensor] = None) -> float:
        """선호도 모델 업데이트"""
        if not self.training_mode:
            return 0.0
            
        preference_loss = 0.0
        
        # 명시적 선호도 처리
        if preferences is not None:
            for key, pref_value in preferences.items():
                if key in outputs:
                    output_data = outputs[key]
                    if isinstance(output_data, torch.Tensor):
                        # 텐서 차원 조정
                        flattened_data = self._prepare_input_data(output_data)
                        
                        # 예측 및 손실 계산
                        prediction, confidence = self.forward(flattened_data)
                        
                        target = torch.tensor([pref_value], dtype=torch.float32, device=self.device)
                        target = target.expand(prediction.size(0))
                        loss = F.mse_loss(prediction.squeeze(), target)
                        preference_loss += loss.item()
                        
                        # 메모리에 저장
                        self._store_preference_data(key, flattened_data, pref_value, 'explicit')
                        
                        # 학습기 업데이트
                        context = {
                            'confidence': confidence.mean().item(),
                            'reward': rewards.mean().item() if rewards is not None else 0.0,
                            'time_since_last_update': 1.0  # 간단한 시간 정보
                        }
                        
                        self.learner.update_preference(
                            prediction.squeeze(), target, 'online', context
                        )
                        
        # 암시적 선호도 처리 (보상 기반)
        if rewards is not None:
            implicit_preferences = self._rewards_to_preferences(rewards, outputs)
            for key, pref_value in implicit_preferences.items():
                if key in outputs:
                    output_data = outputs[key]
                    if isinstance(output_data, torch.Tensor):
                        flattened_data = self._prepare_input_data(output_data)
                        self._store_preference_data(key, flattened_data, pref_value, 'implicit')
                        
        self.update_count += 1
        return preference_loss
        
    def _prepare_input_data(self, output_data: torch.Tensor) -> torch.Tensor:
        """입력 데이터 준비"""
        # 불린 텐서를 float로 변환
        if output_data.dtype == torch.bool:
            output_data = output_data.float()
            
        # 텐서 차원 조정
        if output_data.dim() > 2:
            # 3D 이상 텐서를 2D로 평탄화
            batch_size = output_data.size(0)
            flattened_data = output_data.view(batch_size, -1)
        else:
            flattened_data = output_data
            
        # 입력 차원 확인 및 조정
        if flattened_data.size(-1) != self.input_dim:
            # 차원이 맞지 않으면 평균 풀링으로 조정
            if flattened_data.size(-1) > self.input_dim:
                # 차원 축소
                flattened_data = F.adaptive_avg_pool1d(
                    flattened_data.unsqueeze(1), 
                    self.input_dim
                ).squeeze(1)
            else:
                # 차원 확장 (패딩)
                padding_size = self.input_dim - flattened_data.size(-1)
                flattened_data = F.pad(flattened_data, (0, padding_size))
                
        return flattened_data
        
    def _store_preference_data(self, key: str, input_data: torch.Tensor, preference_value: float, preference_type: str):
        """선호도 데이터 저장"""
        preference_data = PreferenceData(
            preference_type=preference_type,
            input_data=input_data,
            preference_value=preference_value,
            timestamp=time.time(),
            context={'key': key},
            confidence=0.8  # 기본 신뢰도
        )
        
        self.memory.store(preference_data, key)
        
    def _rewards_to_preferences(self, rewards: torch.Tensor, outputs: Dict[str, Any]) -> Dict[str, float]:
        """보상을 선호도로 변환"""
        preferences = {}
        avg_reward = rewards.mean().item()
        
        # 출력의 각 구성 요소에 대해 선호도 할당
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 0:
                    # 불린 텐서 처리
                    if value.dtype == torch.bool:
                        # 불린 텐서를 float로 변환
                        value_float = value.float()
                        mean_val = value_float.mean().item()
                        var_val = value_float.var().item()
                        max_val = value_float.max().item()
                    else:
                        # 일반 텐서 처리
                        mean_val = value.mean().item()
                        var_val = value.var().item()
                        max_val = value.max().item()
                    
                    # 복합 선호도 점수 계산
                    preference_score = (
                        0.4 * (mean_val / max(1, max_val)) +
                        0.3 * (1.0 / max(1, var_val)) +
                        0.3 * (avg_reward / max(1, abs(avg_reward)))
                    )
                    
                    preferences[key] = np.clip(preference_score, 0.0, 1.0)
                    
        return preferences
        
    def get_preference_summary(self) -> Dict[str, Any]:
        """선호도 모델 요약 정보"""
        return {
            'update_count': self.update_count,
            'training_mode': self.training_mode,
            'model_params': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim
            },
            'learner_stats': self.learner.get_learning_stats(),
            'memory_stats': self.memory.get_memory_stats()
        }
        
    def train(self, mode: bool = True):
        """훈련 모드 설정"""
        super().train(mode)
        self.training_mode = mode
        
    def eval(self):
        """평가 모드 설정"""
        super().eval()
        self.training_mode = False
