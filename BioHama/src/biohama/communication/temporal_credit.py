"""
시간적 신용 할당 모듈

BioHama 시스템의 장기 의존성 학습을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque

from ..core.base.module_interface import ModuleInterface


class TemporalCredit(ModuleInterface):
    """
    시간적 신용 할당 시스템
    
    장기 의존성을 고려한 신용 할당을 수행합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        시간적 신용 할당 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 시간적 신용 설정
        self.credit_decay = config.get('credit_decay', 0.9)
        self.max_temporal_window = config.get('max_temporal_window', 50)
        self.eligibility_trace_decay = config.get('eligibility_trace_decay', 0.95)
        
        # 신용 할당 네트워크
        self.credit_network = nn.Sequential(
            nn.Linear(self.config.get('input_dim', 128), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 시간적 메모리
        self.temporal_memory = deque(maxlen=self.max_temporal_window)
        self.eligibility_traces = {}
        self.credit_history = {}
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """시간적 신용 할당 순전파"""
        module_id = inputs.get('module_id')
        action = inputs.get('action')
        reward = inputs.get('reward', 0.0)
        action_type = inputs.get('action_type', 'update')
        
        if action_type == 'update' and module_id and action is not None:
            # 시간적 메모리에 추가
            self._add_to_memory(module_id, action, reward)
            
            # 신용 할당 계산
            credit_assignment = self._compute_credit_assignment(module_id, reward)
            
            return {
                'module_id': module_id,
                'credit_assignment': credit_assignment,
                'success': True
            }
        
        elif action_type == 'get_credit':
            # 신용 정보 반환
            return self._get_credit_info(inputs.get('module_id'))
        
        else:
            # 전체 신용 상태 반환
            return self._get_global_credit_state()
    
    def _add_to_memory(self, module_id: str, action: Any, reward: float) -> None:
        """시간적 메모리에 추가"""
        memory_entry = {
            'module_id': module_id,
            'action': action,
            'reward': reward,
            'timestamp': len(self.temporal_memory)
        }
        
        self.temporal_memory.append(memory_entry)
        
        # 적격성 흔적 업데이트
        self._update_eligibility_trace(module_id)
    
    def _update_eligibility_trace(self, module_id: str) -> None:
        """적격성 흔적 업데이트"""
        # 모든 모듈의 적격성 흔적 감쇠
        for mid in self.eligibility_traces:
            self.eligibility_traces[mid] *= self.eligibility_trace_decay
        
        # 현재 모듈의 적격성 흔적 증가
        if module_id not in self.eligibility_traces:
            self.eligibility_traces[module_id] = 0.0
        
        self.eligibility_traces[module_id] += 1.0
    
    def _compute_credit_assignment(self, module_id: str, reward: float) -> Dict[str, float]:
        """신용 할당 계산"""
        credit_assignment = {}
        
        # 시간적 윈도우 내의 모든 모듈에 신용 할당
        for entry in self.temporal_memory:
            entry_module_id = entry['module_id']
            time_diff = len(self.temporal_memory) - entry['timestamp']
            
            # 시간적 감쇠
            temporal_decay = self.credit_decay ** time_diff
            
            # 적격성 흔적
            eligibility = self.eligibility_traces.get(entry_module_id, 0.0)
            
            # 신용 계산
            credit = reward * temporal_decay * eligibility
            
            if entry_module_id not in credit_assignment:
                credit_assignment[entry_module_id] = 0.0
            
            credit_assignment[entry_module_id] += credit
        
        # 신용 히스토리 업데이트
        for mid, credit in credit_assignment.items():
            if mid not in self.credit_history:
                self.credit_history[mid] = []
            
            self.credit_history[mid].append(credit)
            
            # 히스토리 크기 제한
            if len(self.credit_history[mid]) > 100:
                self.credit_history[mid] = self.credit_history[mid][-100:]
        
        return credit_assignment
    
    def _get_credit_info(self, module_id: str) -> Dict[str, Any]:
        """특정 모듈의 신용 정보 반환"""
        if not module_id:
            return {}
        
        credit_history = self.credit_history.get(module_id, [])
        eligibility = self.eligibility_traces.get(module_id, 0.0)
        
        return {
            'module_id': module_id,
            'current_eligibility': eligibility,
            'avg_credit': np.mean(credit_history) if credit_history else 0.0,
            'total_credit': sum(credit_history),
            'credit_history_length': len(credit_history)
        }
    
    def _get_global_credit_state(self) -> Dict[str, Any]:
        """전체 신용 상태 반환"""
        return {
            'temporal_memory_size': len(self.temporal_memory),
            'active_modules': len(self.eligibility_traces),
            'total_credit_assignments': len(self.credit_history),
            'eligibility_traces': self.eligibility_traces.copy()
        }
    
    def get_top_credited_modules(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """상위 신용 모듈 반환"""
        module_credits = []
        
        for module_id, credit_history in self.credit_history.items():
            avg_credit = np.mean(credit_history) if credit_history else 0.0
            module_credits.append((module_id, avg_credit))
        
        # 신용 순으로 정렬
        module_credits.sort(key=lambda x: x[1], reverse=True)
        
        return module_credits[:top_k]
    
    def clear_temporal_memory(self) -> None:
        """시간적 메모리 초기화"""
        self.temporal_memory.clear()
        self.eligibility_traces.clear()
    
    def reset(self) -> None:
        """시간적 신용 할당 시스템 초기화"""
        super().reset()
        self.temporal_memory.clear()
        self.eligibility_traces.clear()
        self.credit_history.clear()

