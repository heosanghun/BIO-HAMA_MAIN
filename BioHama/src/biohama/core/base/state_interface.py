"""
상태 인터페이스 정의

BioHama 시스템의 상태 관리를 위한 기본 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np


class StateInterface(ABC):
    """
    상태 관리의 기본 인터페이스
    
    이 인터페이스는 시스템의 다양한 상태를 관리하고
    상태 간 전환을 처리하는 메커니즘을 정의합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        상태 관리자 초기화
        
        Args:
            config: 상태 관리 설정 딕셔너리
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.state_dim = config.get('state_dim', 128)
        self.device = config.get('device', 'cpu')
        
        # 상태 저장소
        self.current_state = {}
        self.state_history = []
        self.max_history_length = config.get('max_history_length', 1000)
        
    @abstractmethod
    def update_state(self, new_state: Dict[str, Any], 
                    transition_info: Optional[Dict[str, Any]] = None) -> None:
        """
        상태 업데이트
        
        Args:
            new_state: 새로운 상태 정보
            transition_info: 상태 전환 정보 (선택사항)
        """
        pass
    
    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """
        현재 상태 반환
        
        Returns:
            현재 상태 딕셔너리
        """
        pass
    
    @abstractmethod
    def get_state_at_time(self, timestamp: Union[int, float]) -> Optional[Dict[str, Any]]:
        """
        특정 시점의 상태 반환
        
        Args:
            timestamp: 시간 스탬프
            
        Returns:
            해당 시점의 상태 (없으면 None)
        """
        pass
    
    def add_state_observer(self, observer: Any) -> None:
        """
        상태 관찰자 추가
        
        Args:
            observer: 상태 변화를 관찰할 객체
        """
        if not hasattr(self, 'observers'):
            self.observers = []
        self.observers.append(observer)
    
    def remove_state_observer(self, observer: Any) -> None:
        """
        상태 관찰자 제거
        
        Args:
            observer: 제거할 관찰자 객체
        """
        if hasattr(self, 'observers'):
            self.observers = [obs for obs in self.observers if obs != observer]
    
    def notify_observers(self, state_change: Dict[str, Any]) -> None:
        """
        관찰자들에게 상태 변화 알림
        
        Args:
            state_change: 상태 변화 정보
        """
        if hasattr(self, 'observers'):
            for observer in self.observers:
                if hasattr(observer, 'on_state_change'):
                    observer.on_state_change(state_change)
    
    def get_state_history(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        상태 히스토리 반환
        
        Args:
            length: 반환할 히스토리 길이 (None이면 전체)
            
        Returns:
            상태 히스토리 목록
        """
        if length is None:
            return self.state_history.copy()
        return self.state_history[-length:]
    
    def clear_state_history(self) -> None:
        """상태 히스토리 초기화"""
        self.state_history = []
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        상태 통계 정보 반환
        
        Returns:
            상태 통계 딕셔너리
        """
        if not self.state_history:
            return {}
        
        stats = {
            'total_states': len(self.state_history),
            'state_keys': list(self.current_state.keys()) if self.current_state else [],
            'history_length': len(self.state_history)
        }
        
        # 상태 변화 빈도 분석
        if len(self.state_history) > 1:
            changes = []
            for i in range(1, len(self.state_history)):
                prev_state = self.state_history[i-1]
                curr_state = self.state_history[i]
                change_count = sum(1 for key in curr_state 
                                 if key not in prev_state or curr_state[key] != prev_state[key])
                changes.append(change_count)
            
            stats['avg_state_changes'] = np.mean(changes) if changes else 0
            stats['max_state_changes'] = max(changes) if changes else 0
        
        return stats
    
    def save_state(self, filepath: str) -> None:
        """
        상태를 파일에 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        import pickle
        state_data = {
            'current_state': self.current_state,
            'state_history': self.state_history,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
    
    def load_state(self, filepath: str) -> None:
        """
        파일에서 상태 로드
        
        Args:
            filepath: 로드할 파일 경로
        """
        import pickle
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        self.current_state = state_data.get('current_state', {})
        self.state_history = state_data.get('state_history', [])
        self.config.update(state_data.get('config', {}))
    
    def reset(self) -> None:
        """상태 관리자 초기화"""
        self.current_state = {}
        self.state_history = []
        
    def __str__(self) -> str:
        """상태 관리자 문자열 표현"""
        return f"{self.__class__.__name__}(dim={self.state_dim})"
    
    def __repr__(self) -> str:
        """상태 관리자 상세 문자열 표현"""
        return f"{self.__class__.__name__}(name={self.name}, dim={self.state_dim})"
