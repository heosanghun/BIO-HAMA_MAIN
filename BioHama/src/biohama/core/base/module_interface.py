"""
모듈 인터페이스 정의

모든 BioHama 인지 모듈이 구현해야 하는 기본 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ModuleInterface(ABC):
    """
    모든 BioHama 인지 모듈의 기본 인터페이스
    
    이 인터페이스는 모듈의 기본 동작을 정의하며,
    모든 인지 모듈은 이 인터페이스를 구현해야 합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        모듈 초기화
        
        Args:
            config: 모듈 설정 딕셔너리
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.module_id = config.get('module_id', None)
        self.is_active = config.get('is_active', True)
        self.device = config.get('device', 'cpu')
        
        # 모듈 상태
        self.state = {}
        self.metadata = {}
        
    @abstractmethod
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        모듈의 순전파 처리
        
        Args:
            inputs: 입력 데이터 딕셔너리
            context: 컨텍스트 정보 (선택사항)
            
        Returns:
            출력 데이터 딕셔너리
        """
        pass
    
    @abstractmethod
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        모듈 상태 업데이트
        
        Args:
            new_state: 새로운 상태 정보
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        현재 모듈 상태 반환
        
        Returns:
            현재 상태 딕셔너리
        """
        pass
    
    def activate(self) -> None:
        """모듈 활성화"""
        self.is_active = True
        
    def deactivate(self) -> None:
        """모듈 비활성화"""
        self.is_active = False
        
    def is_ready(self) -> bool:
        """
        모듈이 처리 준비가 되었는지 확인
        
        Returns:
            준비 상태 여부
        """
        return self.is_active
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        모듈 메타데이터 반환
        
        Returns:
            메타데이터 딕셔너리
        """
        return self.metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        메타데이터 설정
        
        Args:
            key: 메타데이터 키
            value: 메타데이터 값
        """
        self.metadata[key] = value
    
    def reset(self) -> None:
        """모듈 상태 초기화"""
        self.state = {}
        self.metadata = {}
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        모듈 파라미터 반환
        
        Returns:
            파라미터 딕셔너리
        """
        if hasattr(self, 'parameters'):
            return {name: param.data.clone() for name, param in self.named_parameters()}
        return {}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        모듈 파라미터 설정
        
        Args:
            parameters: 파라미터 딕셔너리
        """
        if hasattr(self, 'named_parameters'):
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
    
    def get_complexity(self) -> Dict[str, float]:
        """
        모듈 복잡도 정보 반환
        
        Returns:
            복잡도 정보 딕셔너리 (파라미터 수, 연산량 등)
        """
        total_params = 0
        trainable_params = 0
        
        if hasattr(self, 'parameters'):
            for param in self.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """모듈 문자열 표현"""
        return f"{self.__class__.__name__}(name={self.name}, active={self.is_active})"
    
    def __repr__(self) -> str:
        """모듈 상세 문자열 표현"""
        return f"{self.__class__.__name__}(name={self.name}, id={self.module_id}, active={self.is_active})"
