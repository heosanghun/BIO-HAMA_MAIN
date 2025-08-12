"""
헤비안 학습 모듈

BioHama 시스템의 연결 강화 학습을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.base.module_interface import ModuleInterface


class HebbianLearning(ModuleInterface):
    """
    헤비안 학습 시스템
    
    "함께 활성화되는 뉴런들은 함께 연결된다"는 헤비안 학습 규칙을 구현합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        헤비안 학습 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 헤비안 학습 설정
        self.learning_rate = config.get('learning_rate', 0.01)
        self.decay_rate = config.get('decay_rate', 0.99)
        self.connection_threshold = config.get('connection_threshold', 0.1)
        
        # 연결 가중치 행렬
        self.connection_matrix = {}
        self.activation_history = {}
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """헤비안 학습 순전파"""
        module_id = inputs.get('module_id')
        activation = inputs.get('activation')
        action = inputs.get('action', 'update')
        
        if action == 'update' and module_id and activation is not None:
            # 활성화 기록 업데이트
            self._update_activation(module_id, activation)
            
            # 헤비안 학습 수행
            connection_updates = self._hebbian_update(module_id, activation)
            
            return {
                'module_id': module_id,
                'connection_updates': connection_updates,
                'success': True
            }
        
        elif action == 'get_connections':
            # 연결 정보 반환
            return self._get_connection_info(inputs.get('module_id'))
        
        else:
            # 전체 연결 상태 반환
            return self._get_global_state()
    
    def _update_activation(self, module_id: str, activation: float) -> None:
        """활성화 기록 업데이트"""
        if module_id not in self.activation_history:
            self.activation_history[module_id] = []
        
        self.activation_history[module_id].append(activation)
        
        # 히스토리 크기 제한
        if len(self.activation_history[module_id]) > 100:
            self.activation_history[module_id] = self.activation_history[module_id][-100:]
    
    def _hebbian_update(self, module_id: str, activation: float) -> Dict[str, float]:
        """헤비안 학습 업데이트"""
        connection_updates = {}
        
        # 다른 모듈들과의 연결 업데이트
        for other_id in self.activation_history:
            if other_id != module_id:
                # 상관관계 계산
                correlation = self._compute_correlation(module_id, other_id)
                
                # 연결 가중치 업데이트
                connection_key = (module_id, other_id)
                current_weight = self.connection_matrix.get(connection_key, 0.0)
                
                # 헤비안 규칙: 상관관계가 높을수록 연결 강화
                new_weight = current_weight + self.learning_rate * correlation * activation
                
                # 감쇠 적용
                new_weight *= self.decay_rate
                
                # 임계값 적용
                if abs(new_weight) < self.connection_threshold:
                    new_weight = 0.0
                
                self.connection_matrix[connection_key] = new_weight
                connection_updates[other_id] = new_weight
        
        return connection_updates
    
    def _compute_correlation(self, module_id1: str, module_id2: str) -> float:
        """두 모듈 간의 상관관계 계산"""
        if (module_id1 not in self.activation_history or 
            module_id2 not in self.activation_history):
            return 0.0
        
        activations1 = self.activation_history[module_id1]
        activations2 = self.activation_history[module_id2]
        
        # 최소 길이로 맞춤
        min_length = min(len(activations1), len(activations2))
        if min_length < 2:
            return 0.0
        
        activations1 = activations1[-min_length:]
        activations2 = activations2[-min_length:]
        
        # 상관관계 계산
        try:
            correlation = np.corrcoef(activations1, activations2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _get_connection_info(self, module_id: str) -> Dict[str, Any]:
        """특정 모듈의 연결 정보 반환"""
        if not module_id:
            return {}
        
        connections = {}
        for (source, target), weight in self.connection_matrix.items():
            if source == module_id:
                connections[target] = weight
            elif target == module_id:
                connections[source] = weight
        
        return {
            'module_id': module_id,
            'connections': connections,
            'activation_level': np.mean(self.activation_history.get(module_id, [0.0]))
        }
    
    def _get_global_state(self) -> Dict[str, Any]:
        """전체 연결 상태 반환"""
        return {
            'total_connections': len(self.connection_matrix),
            'active_modules': len(self.activation_history),
            'connection_matrix': self.connection_matrix.copy(),
            'activation_history': {k: len(v) for k, v in self.activation_history.items()}
        }
    
    def get_strongest_connections(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """가장 강한 연결들 반환"""
        sorted_connections = sorted(
            self.connection_matrix.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return [(source, target, weight) for (source, target), weight in sorted_connections[:top_k]]
    
    def prune_weak_connections(self, threshold: float = None) -> int:
        """약한 연결 제거"""
        if threshold is None:
            threshold = self.connection_threshold
        
        pruned_count = 0
        connections_to_remove = []
        
        for (source, target), weight in self.connection_matrix.items():
            if abs(weight) < threshold:
                connections_to_remove.append((source, target))
                pruned_count += 1
        
        for connection in connections_to_remove:
            del self.connection_matrix[connection]
        
        return pruned_count
    
    def reset(self) -> None:
        """헤비안 학습 시스템 초기화"""
        super().reset()
        self.connection_matrix.clear()
        self.activation_history.clear()

