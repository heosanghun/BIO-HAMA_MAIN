"""
정책 최적화 모듈

BioHama 시스템의 강화학습 기반 정책 최적화를 담당합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base.module_interface import ModuleInterface


class PolicyOptimizer(ModuleInterface):
    """
    정책 최적화기
    
    강화학습 기반으로 정책을 최적화합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        정책 최적화기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 최적화 설정
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 정책 네트워크
        self.policy_network = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 256),
            nn.ReLU(),
            nn.Linear(256, self.config.get('action_dim', 64))
        )
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """정책 최적화기 순전파"""
        state = inputs.get('state')
        if state is None:
            return {'error': 'State not provided'}
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        policy_logits = self.policy_network(state)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        return {
            'policy_logits': policy_logits,
            'action_probs': action_probs
        }
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                     advantages: torch.Tensor, old_log_probs: torch.Tensor) -> Dict[str, float]:
        """정책 업데이트"""
        policy_logits = self.policy_network(states)
        action_probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions)
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def reset(self) -> None:
        """정책 최적화기 초기화"""
        super().reset()

