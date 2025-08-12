"""
의사결정 엔진 모듈

BioHama 시스템의 고차 인지 의사결정을 담당하는 모듈입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base.state_interface import StateInterface
from .base.module_interface import ModuleInterface


class DecisionEngine(StateInterface, ModuleInterface, nn.Module):
    """
    의사결정 엔진
    
    시스템의 고차 인지 의사결정을 처리합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        의사결정 엔진 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        StateInterface.__init__(self, config)
        ModuleInterface.__init__(self, config)
        nn.Module.__init__(self)
        
        # 의사결정 설정
        self.decision_dim = config.get('decision_dim', 256)
        self.num_options = config.get('num_options', 10)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        # 의사결정 구성 요소들
        self._initialize_decision_components()
        
    def _initialize_decision_components(self):
        """의사결정 구성 요소들을 초기화합니다."""
        
        # 의사결정 네트워크
        self.decision_network = nn.Sequential(
            nn.Linear(self.decision_dim, self.decision_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.decision_dim // 2, self.num_options)
        )
        
        # 현재 상태 설정
        self.current_state = {
            'decision_history': [],
            'confidence_level': 0.5,
            'exploration_mode': False,
            'timestamp': None
        }
        
    def make_decision(self, context: torch.Tensor, options: List[Any], 
                     constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        의사결정 수행
        
        Args:
            context: 컨텍스트 정보
            options: 선택 가능한 옵션들
            constraints: 제약 조건 (선택사항)
            
        Returns:
            의사결정 결과
        """
        # 컨텍스트 처리
        if context.size(0) != self.decision_dim:
            context = F.linear(context, torch.randn(self.decision_dim, context.size(0)))
        
        # 의사결정 점수 계산
        decision_scores = self.decision_network(context)
        
        # 제약 조건 적용
        if constraints:
            decision_scores = self._apply_constraints(decision_scores, constraints)
        
        # 탐색 vs 활용 결정
        if torch.rand(1) < self.exploration_rate:
            selected_idx = torch.randint(0, len(options), (1,))
            exploration = True
        else:
            selected_idx = torch.argmax(decision_scores)
            exploration = False
        
        # 결과 생성
        decision_result = {
            'selected_option': options[selected_idx.item()],
            'selected_index': selected_idx.item(),
            'confidence': F.softmax(decision_scores, dim=0)[selected_idx].item(),
            'exploration': exploration,
            'all_scores': decision_scores.tolist(),
            'context': context.tolist()
        }
        
        # 의사결정 히스토리에 추가
        self.current_state['decision_history'].append(decision_result)
        
        return decision_result
    
    def _apply_constraints(self, scores: torch.Tensor, 
                          constraints: Dict[str, Any]) -> torch.Tensor:
        """
        제약 조건 적용
        
        Args:
            scores: 원본 점수
            constraints: 제약 조건
            
        Returns:
            제약 조건이 적용된 점수
        """
        # 간단한 제약 조건 적용 (마스킹)
        if 'mask' in constraints:
            mask = constraints['mask']
            scores = scores * mask
        
        # 가중치 적용
        if 'weights' in constraints:
            weights = constraints['weights']
            scores = scores * weights
        
        return scores
    
    def update_decision_policy(self, feedback: Dict[str, Any]) -> None:
        """
        의사결정 정책 업데이트
        
        Args:
            feedback: 피드백 정보
        """
        # 피드백 기반 정책 업데이트 (간단한 구현)
        if 'reward' in feedback:
            reward = feedback['reward']
            # 긍정적 피드백이면 탐색률 감소
            if reward > 0.5:
                self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
            else:
                # 부정적 피드백이면 탐색률 증가
                self.exploration_rate = min(0.3, self.exploration_rate * 1.05)
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """
        의사결정 통계 반환
        
        Returns:
            의사결정 통계 딕셔너리
        """
        if not self.current_state['decision_history']:
            return {
                'total_decisions': 0,
                'exploration_rate': self.exploration_rate,
                'avg_confidence': 0.0
            }
        
        history = self.current_state['decision_history']
        confidences = [decision['confidence'] for decision in history]
        exploration_count = sum(1 for decision in history if decision['exploration'])
        
        return {
            'total_decisions': len(history),
            'exploration_rate': exploration_count / len(history),
            'avg_confidence': np.mean(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences)
        }
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """상태 업데이트"""
        self.current_state.update(new_state)
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.current_state.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """ModuleInterface 요구사항: 현재 상태 반환"""
        return self.current_state.copy()
    
    def get_state_at_time(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """특정 시간의 상태 반환 (기본 구현)"""
        # 간단한 구현: 현재 상태 반환
        return self.current_state.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            'name': getattr(self, 'name', 'DecisionEngine'),
            'module_type': 'decision_engine'
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """모듈 복잡도를 반환합니다."""
        total_params = sum(p.numel() for p in self.decision_network.parameters()) if hasattr(self, 'decision_network') else 0
        trainable_params = sum(p.numel() for p in self.decision_network.parameters() if p.requires_grad) if hasattr(self, 'decision_network') else 0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': 'decision_engine'
        }
    
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        의사결정 엔진 순전파
        
        Args:
            inputs: 입력 데이터
            context: 컨텍스트 정보
            
        Returns:
            의사결정 결과
        """
        # 입력에서 컨텍스트 추출
        if 'context' in inputs:
            context_tensor = torch.tensor(inputs['context'], dtype=torch.float32)
        else:
            # 기본 컨텍스트 생성
            context_tensor = torch.randn(self.decision_dim)
        
        # 옵션 추출 또는 생성
        options = inputs.get('options', list(range(self.num_options)))
        
        # 제약 조건 추출
        constraints = inputs.get('constraints', None)
        
        # 의사결정 수행
        decision_result = self.make_decision(context_tensor, options, constraints)
        
        return {
            'decision': decision_result,
            'confidence': decision_result['confidence'],
            'exploration': decision_result['exploration']
        }
    
    def reset(self) -> None:
        """의사결정 엔진 초기화"""
        super().reset()
        self._initialize_decision_components()
