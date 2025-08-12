"""
메타 학습 모듈

BioHama 시스템의 빠른 적응을 위한 메타 학습을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base.module_interface import ModuleInterface


class MetaLearning(ModuleInterface):
    """
    메타 학습 시스템
    
    새로운 작업에 빠르게 적응하기 위한 메타 학습을 수행합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        메타 학습 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 메타 학습 설정
        self.meta_learning_rate = config.get('meta_learning_rate', 1e-3)
        self.inner_learning_rate = config.get('inner_learning_rate', 1e-2)
        self.num_inner_steps = config.get('num_inner_steps', 5)
        
        # 메타 네트워크
        self.meta_network = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.get('action_dim', 64))
        )
        
        # 메타 옵티마이저
        self.meta_optimizer = torch.optim.Adam(self.meta_network.parameters(), lr=self.meta_learning_rate)
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """메타 학습 순전파"""
        state = inputs.get('state')
        if state is None:
            return {'error': 'State not provided'}
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # 메타 네트워크로 행동 예측
        action_logits = self.meta_network(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs
        }
    
    def adapt_to_task(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        새로운 작업에 적응
        
        Args:
            task_data: 작업 데이터
            
        Returns:
            적응 결과
        """
        # 내부 루프 (작업별 적응)
        adapted_params = self._inner_loop(task_data)
        
        # 외부 루프 (메타 업데이트)
        meta_loss = self._outer_loop(task_data, adapted_params)
        
        return {
            'adapted_params': adapted_params,
            'meta_loss': meta_loss
        }
    
    def _inner_loop(self, task_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """내부 루프 (작업별 적응)"""
        # 현재 파라미터 복사
        adapted_params = {}
        for name, param in self.meta_network.named_parameters():
            adapted_params[name] = param.clone()
        
        # 내부 스텝들
        for step in range(self.num_inner_steps):
            # 작업 데이터로 손실 계산
            loss = self._compute_task_loss(task_data, adapted_params)
            
            # 그래디언트 계산 및 파라미터 업데이트
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_learning_rate * grad
        
        return adapted_params
    
    def _outer_loop(self, task_data: List[Dict[str, Any]], 
                   adapted_params: Dict[str, torch.Tensor]) -> float:
        """외부 루프 (메타 업데이트)"""
        # 적응된 파라미터로 메타 손실 계산
        meta_loss = self._compute_meta_loss(task_data, adapted_params)
        
        # 메타 옵티마이저 스텝
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _compute_task_loss(self, task_data: List[Dict[str, Any]], 
                          params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """작업 손실 계산"""
        total_loss = 0.0
        
        for data in task_data:
            state = torch.tensor(data['state'], dtype=torch.float32)
            target_action = torch.tensor(data['action'], dtype=torch.long)
            
            # 임시 네트워크 생성
            temp_network = self._create_temp_network(params)
            action_logits = temp_network(state)
            
            # 교차 엔트로피 손실
            loss = F.cross_entropy(action_logits, target_action)
            total_loss += loss
        
        return total_loss / len(task_data)
    
    def _compute_meta_loss(self, task_data: List[Dict[str, Any]], 
                          adapted_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """메타 손실 계산"""
        # 적응된 파라미터로 검증 손실 계산
        return self._compute_task_loss(task_data, adapted_params)
    
    def _create_temp_network(self, params: Dict[str, torch.Tensor]) -> nn.Module:
        """임시 네트워크 생성"""
        temp_network = type(self.meta_network)(
            *self.meta_network.children()
        )
        
        # 파라미터 설정
        for name, param in temp_network.named_parameters():
            if name in params:
                param.data = params[name].data
        
        return temp_network
    
    def reset(self) -> None:
        """메타 학습 시스템 초기화"""
        super().reset()

