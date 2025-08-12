"""
라우터 인터페이스 정의

BioHama 시스템의 라우팅 메커니즘을 위한 기본 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch


class RouterInterface(ABC):
    """
    라우터의 기본 인터페이스
    
    이 인터페이스는 입력을 적절한 모듈로 라우팅하는
    메커니즘을 정의합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        라우터 초기화
        
        Args:
            config: 라우터 설정 딕셔너리
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.routing_strategy = config.get('routing_strategy', 'attention')
        self.device = config.get('device', 'cpu')
        
        # 라우팅 상태
        self.routing_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def route(self, inputs: Dict[str, Any], available_modules: List[Any], 
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        입력을 적절한 모듈로 라우팅
        
        Args:
            inputs: 입력 데이터
            available_modules: 사용 가능한 모듈 목록
            context: 컨텍스트 정보 (선택사항)
            
        Returns:
            라우팅 결과 딕셔너리
        """
        pass
    
    @abstractmethod
    def update_routing_weights(self, feedback: Dict[str, Any]) -> None:
        """
        라우팅 가중치 업데이트
        
        Args:
            feedback: 피드백 정보
        """
        pass
    
    @abstractmethod
    def get_routing_confidence(self, module_id: str) -> float:
        """
        특정 모듈에 대한 라우팅 신뢰도 반환
        
        Args:
            module_id: 모듈 ID
            
        Returns:
            라우팅 신뢰도 (0.0 ~ 1.0)
        """
        pass
    
    def add_routing_rule(self, rule: Dict[str, Any]) -> None:
        """
        라우팅 규칙 추가
        
        Args:
            rule: 라우팅 규칙 딕셔너리
        """
        if not hasattr(self, 'routing_rules'):
            self.routing_rules = []
        self.routing_rules.append(rule)
    
    def remove_routing_rule(self, rule_id: str) -> None:
        """
        라우팅 규칙 제거
        
        Args:
            rule_id: 규칙 ID
        """
        if hasattr(self, 'routing_rules'):
            self.routing_rules = [rule for rule in self.routing_rules 
                                if rule.get('id') != rule_id]
    
    def get_routing_history(self) -> List[Dict[str, Any]]:
        """
        라우팅 히스토리 반환
        
        Returns:
            라우팅 히스토리 목록
        """
        return self.routing_history
    
    def clear_routing_history(self) -> None:
        """라우팅 히스토리 초기화"""
        self.routing_history = []
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        성능 지표 반환
        
        Returns:
            성능 지표 딕셔너리
        """
        return self.performance_metrics
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        성능 지표 업데이트
        
        Args:
            metrics: 새로운 성능 지표
        """
        self.performance_metrics.update(metrics)
    
    def reset(self) -> None:
        """라우터 상태 초기화"""
        self.routing_history = []
        self.performance_metrics = {}
        
    def __str__(self) -> str:
        """라우터 문자열 표현"""
        return f"{self.__class__.__name__}(strategy={self.routing_strategy})"
    
    def __repr__(self) -> str:
        """라우터 상세 문자열 표현"""
        return f"{self.__class__.__name__}(name={self.name}, strategy={self.routing_strategy})"
