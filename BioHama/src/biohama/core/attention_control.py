"""
주의 제어 모듈

BioHama 시스템의 주의 메커니즘을 관리하는 모듈입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base.state_interface import StateInterface


class AttentionControl(StateInterface):
    """
    주의 제어 관리자
    
    시스템의 주의 메커니즘을 관리하고 최적화합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        주의 제어 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 주의 설정
        self.attention_dim = config.get('attention_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # 주의 구성 요소들
        self._initialize_attention_components()
        
    def _initialize_attention_components(self):
        """주의 구성 요소들을 초기화합니다."""
        
        # 멀티헤드 어텐션
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 주의 가중치
        self.attention_weights = torch.ones(self.attention_dim)
        
        # 현재 상태 설정
        self.current_state = {
            'attention_weights': self.attention_weights,
            'focus_target': torch.zeros(self.attention_dim),
            'salience_map': torch.zeros(self.attention_dim),
            'distraction_level': 0.0,
            'timestamp': None
        }
        
    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, 
                         value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주의 계산
        
        Args:
            query: 쿼리 텐서
            key: 키 텐서
            value: 값 텐서
            mask: 마스크 (선택사항)
            
        Returns:
            (주의 출력, 주의 가중치)
        """
        attention_output, attention_weights = self.multihead_attention(
            query, key, value, attn_mask=mask
        )
        
        return attention_output, attention_weights
    
    def update_attention_focus(self, focus_target: torch.Tensor, 
                             salience: torch.Tensor) -> None:
        """
        주의 초점 업데이트
        
        Args:
            focus_target: 주의 대상
            salience: 중요도
        """
        alpha = 0.8  # 업데이트 비율
        
        self.current_state['focus_target'] = (
            alpha * self.current_state['focus_target'] + 
            (1 - alpha) * focus_target
        )
        
        self.current_state['salience_map'] = (
            alpha * self.current_state['salience_map'] + 
            (1 - alpha) * salience
        )
        
        # 산만도 계산
        focus_stability = torch.norm(self.current_state['focus_target']).item()
        self.current_state['distraction_level'] = 1.0 - focus_stability
        
    def get_attention_state(self) -> Dict[str, Any]:
        """
        주의 상태 반환
        
        Returns:
            주의 상태 딕셔너리
        """
        return self.current_state.copy()
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """상태 업데이트"""
        self.current_state.update(new_state)
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.current_state.copy()
    
    def get_state_at_time(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """특정 시간의 상태 반환 (기본 구현)"""
        # 간단한 구현: 현재 상태 반환
        return self.current_state.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            'name': getattr(self, 'name', 'AttentionControl'),
            'module_type': 'attention_control'
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """모듈 복잡도를 반환합니다."""
        total_params = sum(p.numel() for p in self.multihead_attention.parameters()) if hasattr(self, 'multihead_attention') else 0
        trainable_params = sum(p.numel() for p in self.multihead_attention.parameters() if p.requires_grad) if hasattr(self, 'multihead_attention') else 0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': 'attention_control'
        }
    
    def reset(self) -> None:
        """주의 제어 초기화"""
        super().reset()
        self._initialize_attention_components()
