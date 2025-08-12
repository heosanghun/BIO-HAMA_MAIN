"""
보상 시스템 모듈

BioHama 시스템의 보상 계산과 관리를 담당합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base.module_interface import ModuleInterface


class RewardSystem(ModuleInterface):
    """
    보상 시스템
    
    다양한 보상 신호를 계산하고 관리합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        보상 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 보상 설정
        self.intrinsic_weight = config.get('intrinsic_weight', 0.1)
        self.extrinsic_weight = config.get('extrinsic_weight', 0.9)
        self.curiosity_weight = config.get('curiosity_weight', 0.05)
        
        # 보상 네트워크
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """보상 시스템 순전파"""
        state = inputs.get('state')
        action = inputs.get('action')
        extrinsic_reward = inputs.get('extrinsic_reward', 0.0)
        
        if state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            # 내재적 보상 계산
            intrinsic_reward = self._compute_intrinsic_reward(state, action)
            
            # 호기심 보상 계산
            curiosity_reward = self._compute_curiosity_reward(state)
            
            # 종합 보상 계산
            total_reward = (
                self.extrinsic_weight * extrinsic_reward +
                self.intrinsic_weight * intrinsic_reward +
                self.curiosity_weight * curiosity_reward
            )
        else:
            total_reward = extrinsic_reward
            intrinsic_reward = 0.0
            curiosity_reward = 0.0
        
        return {
            'total_reward': total_reward,
            'extrinsic_reward': extrinsic_reward,
            'intrinsic_reward': intrinsic_reward,
            'curiosity_reward': curiosity_reward
        }
    
    def _compute_intrinsic_reward(self, state: torch.Tensor, action: Any) -> float:
        """내재적 보상 계산"""
        # 간단한 내재적 보상 (상태 변화 기반)
        if hasattr(self, '_prev_state') and self._prev_state is not None:
            state_change = torch.norm(state - self._prev_state).item()
            intrinsic_reward = min(1.0, state_change)
        else:
            intrinsic_reward = 0.0
        
        self._prev_state = state.clone()
        return intrinsic_reward
    
    def _compute_curiosity_reward(self, state: torch.Tensor) -> float:
        """호기심 보상 계산"""
        # 예측 오차 기반 호기심
        with torch.no_grad():
            predicted_reward = self.reward_predictor(state).item()
            actual_reward = 0.0  # 실제 보상은 외부에서 제공
            prediction_error = abs(predicted_reward - actual_reward)
            curiosity_reward = min(1.0, prediction_error)
        
        return curiosity_reward
    
    def reset(self) -> None:
        """보상 시스템 초기화"""
        super().reset()
        self._prev_state = None

