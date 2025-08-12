"""
인지 상태 관리 모듈

BioHama 시스템의 인지 상태를 관리하고 추적하는 모듈입니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime

from .base.state_interface import StateInterface


class CognitiveState(StateInterface):
    """
    인지 상태 관리자
    
    시스템의 현재 인지 상태를 추적하고 관리합니다.
    작업 메모리, 주의 상태, 감정 상태 등을 포함합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        인지 상태 관리자 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 상태 차원 설정
        self.working_memory_dim = config.get('working_memory_dim', 256)
        self.attention_dim = config.get('attention_dim', 128)
        self.emotion_dim = config.get('emotion_dim', 64)
        self.metacognitive_dim = config.get('metacognitive_dim', 128)
        
        # 상태 구성 요소들
        self._initialize_state_components()
        
        # 상태 전환 히스토리
        self.state_transitions = []
        self.transition_count = 0
        
    def _initialize_state_components(self):
        """상태 구성 요소들을 초기화합니다."""
        
        # 작업 메모리
        self.working_memory = {
            'content': torch.zeros(self.working_memory_dim),
            'capacity': self.working_memory_dim,
            'utilization': 0.0,
            'items': []
        }
        
        # 주의 상태
        self.attention_state = {
            'focus': torch.zeros(self.attention_dim),
            'salience': torch.zeros(self.attention_dim),
            'distraction_level': 0.0,
            'sustained_attention': 1.0
        }
        
        # 감정 상태
        self.emotion_state = {
            'valence': 0.0,  # -1 (부정) ~ 1 (긍정)
            'arousal': 0.0,  # 0 (낮음) ~ 1 (높음)
            'dominance': 0.5,  # 0 (낮음) ~ 1 (높음)
            'emotion_vector': torch.zeros(self.emotion_dim)
        }
        
        # 메타인지 상태
        self.metacognitive_state = {
            'confidence': 0.5,
            'uncertainty': 0.5,
            'self_monitoring': 0.5,
            'metacognitive_vector': torch.zeros(self.metacognitive_dim)
        }
        
        # 인지 부하
        self.cognitive_load = {
            'current_load': 0.0,
            'max_capacity': 1.0,
            'fatigue_level': 0.0,
            'recovery_rate': 0.1
        }
        
        # 현재 상태를 통합 상태로 설정
        self.current_state = {
            'working_memory': self.working_memory,
            'attention': self.attention_state,
            'emotion': self.emotion_state,
            'metacognitive': self.metacognitive_state,
            'cognitive_load': self.cognitive_load,
            'timestamp': datetime.now().isoformat()
        }
        
    def update_state(self, new_state: Dict[str, Any], 
                    transition_info: Optional[Dict[str, Any]] = None) -> None:
        """
        인지 상태 업데이트
        
        Args:
            new_state: 새로운 상태 정보
            transition_info: 상태 전환 정보
        """
        # 이전 상태 저장
        previous_state = self.current_state.copy()
        
        # 상태 구성 요소별 업데이트
        for component, value in new_state.items():
            if component in self.current_state:
                if isinstance(value, dict):
                    self.current_state[component].update(value)
                else:
                    self.current_state[component] = value
        
        # 타임스탬프 업데이트
        self.current_state['timestamp'] = datetime.now().isoformat()
        
        # 상태 히스토리에 추가
        self.state_history.append(previous_state)
        
        # 히스토리 크기 제한
        if len(self.state_history) > self.max_history_length:
            self.state_history = self.state_history[-self.max_history_length:]
        
        # 상태 전환 기록
        if transition_info:
            transition_record = {
                'from_state': previous_state,
                'to_state': self.current_state.copy(),
                'transition_info': transition_info,
                'timestamp': self.current_state['timestamp'],
                'transition_id': self.transition_count
            }
            self.state_transitions.append(transition_record)
            self.transition_count += 1
        
        # 관찰자들에게 알림
        state_change = {
            'previous_state': previous_state,
            'current_state': self.current_state,
            'transition_info': transition_info
        }
        self.notify_observers(state_change)
        
    def get_current_state(self) -> Dict[str, Any]:
        """
        현재 인지 상태 반환
        
        Returns:
            현재 상태 딕셔너리
        """
        return self.current_state.copy()
    
    def get_state_at_time(self, timestamp: Union[int, float]) -> Optional[Dict[str, Any]]:
        """
        특정 시점의 상태 반환
        
        Args:
            timestamp: 시간 스탬프
            
        Returns:
            해당 시점의 상태 (없으면 None)
        """
        # 시간 기반 검색 (간단한 구현)
        for state in reversed(self.state_history):
            if state.get('timestamp') == timestamp:
                return state
        return None
    
    def update_working_memory(self, content: torch.Tensor, 
                            priority: float = 1.0) -> bool:
        """
        작업 메모리 업데이트
        
        Args:
            content: 추가할 내용
            priority: 우선순위
            
        Returns:
            성공 여부
        """
        current_utilization = self.working_memory['utilization']
        
        if current_utilization + priority <= 1.0:
            # 새 항목 추가
            memory_item = {
                'content': content,
                'priority': priority,
                'timestamp': datetime.now().isoformat(),
                'access_count': 0
            }
            
            self.working_memory['items'].append(memory_item)
            self.working_memory['utilization'] += priority
            
            # 메모리 정리 (낮은 우선순위 항목 제거)
            self._cleanup_working_memory()
            
            return True
        else:
            # 메모리 부족 - 낮은 우선순위 항목 제거 후 추가
            if self._evict_low_priority_items(priority):
                return self.update_working_memory(content, priority)
            return False
    
    def _cleanup_working_memory(self):
        """작업 메모리 정리"""
        # 오래된 항목들 제거
        current_time = datetime.now()
        items_to_remove = []
        
        for i, item in enumerate(self.working_memory['items']):
            item_time = datetime.fromisoformat(item['timestamp'])
            time_diff = (current_time - item_time).total_seconds()
            
            # 30초 이상 된 항목은 우선순위 감소
            if time_diff > 30:
                item['priority'] *= 0.9
                
            # 우선순위가 너무 낮은 항목 제거
            if item['priority'] < 0.1:
                items_to_remove.append(i)
        
        # 역순으로 제거 (인덱스 변화 방지)
        for i in reversed(items_to_remove):
            removed_item = self.working_memory['items'].pop(i)
            self.working_memory['utilization'] -= removed_item['priority']
    
    def _evict_low_priority_items(self, required_priority: float) -> bool:
        """낮은 우선순위 항목들을 제거하여 공간 확보"""
        items = self.working_memory['items']
        if not items:
            return False
        
        # 우선순위 순으로 정렬
        items.sort(key=lambda x: x['priority'])
        
        freed_priority = 0.0
        items_to_remove = []
        
        for i, item in enumerate(items):
            if freed_priority >= required_priority:
                break
            freed_priority += item['priority']
            items_to_remove.append(i)
        
        # 역순으로 제거
        for i in reversed(items_to_remove):
            removed_item = items.pop(i)
            self.working_memory['utilization'] -= removed_item['priority']
        
        return freed_priority >= required_priority
    
    def update_attention_state(self, focus_target: torch.Tensor, 
                             salience: torch.Tensor) -> None:
        """
        주의 상태 업데이트
        
        Args:
            focus_target: 주의 대상
            salience: 중요도
        """
        # 주의 초점 업데이트
        alpha = 0.8  # 업데이트 비율
        self.attention_state['focus'] = (
            alpha * self.attention_state['focus'] + 
            (1 - alpha) * focus_target
        )
        
        # 중요도 업데이트
        self.attention_state['salience'] = (
            alpha * self.attention_state['salience'] + 
            (1 - alpha) * salience
        )
        
        # 주의 지속성 계산
        focus_stability = torch.norm(self.attention_state['focus']).item()
        self.attention_state['sustained_attention'] = min(1.0, focus_stability)
        
        # 산만도 계산
        self.attention_state['distraction_level'] = 1.0 - focus_stability
    
    def update_emotion_state(self, valence: float, arousal: float, 
                           dominance: float) -> None:
        """
        감정 상태 업데이트
        
        Args:
            valence: 감정 가치 (-1 ~ 1)
            arousal: 각성 수준 (0 ~ 1)
            dominance: 지배성 (0 ~ 1)
        """
        # 감정 상태 업데이트
        alpha = 0.7  # 감정 변화 속도
        
        self.emotion_state['valence'] = (
            alpha * self.emotion_state['valence'] + 
            (1 - alpha) * valence
        )
        
        self.emotion_state['arousal'] = (
            alpha * self.emotion_state['arousal'] + 
            (1 - alpha) * arousal
        )
        
        self.emotion_state['dominance'] = (
            alpha * self.emotion_state['dominance'] + 
            (1 - alpha) * dominance
        )
        
        # 감정 벡터 업데이트
        emotion_vector = torch.tensor([
            self.emotion_state['valence'],
            self.emotion_state['arousal'],
            self.emotion_state['dominance']
        ])
        
        # 감정 벡터를 고차원으로 확장
        if emotion_vector.size(0) < self.emotion_dim:
            padding = torch.zeros(self.emotion_dim - emotion_vector.size(0))
            emotion_vector = torch.cat([emotion_vector, padding])
        else:
            emotion_vector = emotion_vector[:self.emotion_dim]
        
        self.emotion_state['emotion_vector'] = emotion_vector
    
    def update_metacognitive_state(self, confidence: float, 
                                 uncertainty: float) -> None:
        """
        메타인지 상태 업데이트
        
        Args:
            confidence: 자신감 (0 ~ 1)
            uncertainty: 불확실성 (0 ~ 1)
        """
        alpha = 0.6  # 메타인지 변화 속도
        
        self.metacognitive_state['confidence'] = (
            alpha * self.metacognitive_state['confidence'] + 
            (1 - alpha) * confidence
        )
        
        self.metacognitive_state['uncertainty'] = (
            alpha * self.metacognitive_state['uncertainty'] + 
            (1 - alpha) * uncertainty
        )
        
        # 자기 모니터링 수준 계산
        self.metacognitive_state['self_monitoring'] = (
            self.metacognitive_state['confidence'] * 
            (1 - self.metacognitive_state['uncertainty'])
        )
        
        # 메타인지 벡터 업데이트
        metacognitive_vector = torch.tensor([
            self.metacognitive_state['confidence'],
            self.metacognitive_state['uncertainty'],
            self.metacognitive_state['self_monitoring']
        ])
        
        # 고차원으로 확장
        if metacognitive_vector.size(0) < self.metacognitive_dim:
            padding = torch.zeros(self.metacognitive_dim - metacognitive_vector.size(0))
            metacognitive_vector = torch.cat([metacognitive_vector, padding])
        else:
            metacognitive_vector = metacognitive_vector[:self.metacognitive_dim]
        
        self.metacognitive_state['metacognitive_vector'] = metacognitive_vector
    
    def update_cognitive_load(self, load_increase: float) -> None:
        """
        인지 부하 업데이트
        
        Args:
            load_increase: 부하 증가량
        """
        # 현재 부하 업데이트
        self.cognitive_load['current_load'] = min(
            self.cognitive_load['max_capacity'],
            self.cognitive_load['current_load'] + load_increase
        )
        
        # 피로도 계산
        if self.cognitive_load['current_load'] > 0.8:
            self.cognitive_load['fatigue_level'] = min(
                1.0, 
                self.cognitive_load['fatigue_level'] + 0.1
            )
        else:
            # 회복
            self.cognitive_load['fatigue_level'] = max(
                0.0,
                self.cognitive_load['fatigue_level'] - 
                self.cognitive_load['recovery_rate']
            )
    
    def get_cognitive_state_summary(self) -> Dict[str, Any]:
        """
        인지 상태 요약 반환
        
        Returns:
            상태 요약 딕셔너리
        """
        return {
            'working_memory_utilization': self.working_memory['utilization'],
            'attention_focus': torch.norm(self.attention_state['focus']).item(),
            'attention_distraction': self.attention_state['distraction_level'],
            'emotion_valence': self.emotion_state['valence'],
            'emotion_arousal': self.emotion_state['arousal'],
            'metacognitive_confidence': self.metacognitive_state['confidence'],
            'metacognitive_uncertainty': self.metacognitive_state['uncertainty'],
            'cognitive_load': self.cognitive_load['current_load'],
            'fatigue_level': self.cognitive_load['fatigue_level'],
            'timestamp': self.current_state['timestamp']
        }
    
    def is_cognitive_overload(self) -> bool:
        """
        인지 과부하 상태인지 확인
        
        Returns:
            과부하 여부
        """
        return (
            self.cognitive_load['current_load'] > 0.9 or
            self.cognitive_load['fatigue_level'] > 0.8 or
            self.attention_state['distraction_level'] > 0.7
        )
    
    def get_state_transitions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        상태 전환 히스토리 반환
        
        Args:
            limit: 반환할 전환 수 제한
            
        Returns:
            상태 전환 목록
        """
        if limit is None:
            return self.state_transitions.copy()
        return self.state_transitions[-limit:]
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            'name': getattr(self, 'name', 'CognitiveState'),
            'module_type': 'cognitive_state'
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """모듈 복잡도를 반환합니다."""
        return {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'module_type': 'cognitive_state'
        }
    
    def reset(self) -> None:
        """인지 상태 초기화"""
        super().reset()
        self._initialize_state_components()
        self.state_transitions = []
        self.transition_count = 0
