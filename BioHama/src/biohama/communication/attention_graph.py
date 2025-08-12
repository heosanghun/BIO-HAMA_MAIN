"""
주의 그래프 모듈

BioHama 시스템의 동적 주의 메커니즘을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from ..core.base.module_interface import ModuleInterface


class AttentionGraph(ModuleInterface):
    """
    주의 그래프
    
    동적 주의 메커니즘을 그래프 구조로 관리합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        주의 그래프 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 그래프 설정
        self.max_nodes = config.get('max_nodes', 100)
        self.attention_dim = config.get('attention_dim', 128)
        self.decay_rate = config.get('decay_rate', 0.95)
        
        # 주의 네트워크
        self.attention_network = nn.Sequential(
            nn.Linear(self.config.get('input_dim', 128), self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, self.attention_dim)
        )
        
        # 그래프 구조
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.edge_weights = {}
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """주의 그래프 순전파"""
        node_id = inputs.get('node_id')
        node_features = inputs.get('node_features')
        action = inputs.get('action', 'update')
        
        if action == 'add_node' and node_id and node_features:
            # 노드 추가
            self._add_node(node_id, node_features)
            return {'success': True, 'node_id': node_id}
        
        elif action == 'update_attention':
            # 주의 업데이트
            attention_scores = self._compute_attention_scores()
            return {'attention_scores': attention_scores}
        
        else:
            # 그래프 상태 반환
            return self._get_graph_state()
    
    def _add_node(self, node_id: str, features: Any) -> None:
        """노드 추가"""
        if len(self.graph.nodes) >= self.max_nodes:
            # 가장 오래된 노드 제거
            oldest_node = min(self.graph.nodes, key=lambda x: self.node_features.get(x, {}).get('timestamp', 0))
            self._remove_node(oldest_node)
        
        # 노드 추가
        self.graph.add_node(node_id)
        
        # 특징 인코딩
        if isinstance(features, torch.Tensor):
            encoded_features = self.attention_network(features)
        else:
            encoded_features = torch.randn(self.attention_dim)
        
        # 노드 특징 저장
        self.node_features[node_id] = {
            'features': encoded_features,
            'timestamp': len(self.graph.nodes),
            'attention_level': 1.0
        }
    
    def _remove_node(self, node_id: str) -> None:
        """노드 제거"""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            if node_id in self.node_features:
                del self.node_features[node_id]
    
    def _compute_attention_scores(self) -> Dict[str, float]:
        """주의 점수 계산"""
        attention_scores = {}
        
        for node_id in self.graph.nodes:
            if node_id in self.node_features:
                features = self.node_features[node_id]['features']
                attention_level = self.node_features[node_id]['attention_level']
                
                # 주의 점수 계산 (간단한 구현)
                score = torch.norm(features).item() * attention_level
                attention_scores[node_id] = score
                
                # 주의 수준 감쇠
                self.node_features[node_id]['attention_level'] *= self.decay_rate
        
        return attention_scores
    
    def _get_graph_state(self) -> Dict[str, Any]:
        """그래프 상태 반환"""
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'node_ids': list(self.graph.nodes),
            'edge_list': list(self.graph.edges)
        }
    
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """엣지 추가"""
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, weight=weight)
            self.edge_weights[(source, target)] = weight
    
    def update_attention(self, node_id: str, attention_boost: float) -> None:
        """주의 수준 업데이트"""
        if node_id in self.node_features:
            self.node_features[node_id]['attention_level'] = min(
                1.0, self.node_features[node_id]['attention_level'] + attention_boost
            )
    
    def get_top_attention_nodes(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """상위 주의 노드 반환"""
        attention_scores = self._compute_attention_scores()
        sorted_nodes = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def reset(self) -> None:
        """주의 그래프 초기화"""
        super().reset()
        self.graph.clear()
        self.node_features.clear()
        self.edge_weights.clear()

