"""
작업 메모리 모듈

BioHama 시스템의 작업 메모리를 관리하는 모듈입니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from .base.state_interface import StateInterface


class WorkingMemory(StateInterface):
    """
    작업 메모리 관리자
    
    시스템의 작업 메모리를 관리하고 최적화합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        작업 메모리 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 메모리 설정
        self.capacity = config.get('capacity', 256)
        self.chunk_size = config.get('chunk_size', 64)
        self.decay_rate = config.get('decay_rate', 0.1)
        self.consolidation_threshold = config.get('consolidation_threshold', 0.7)
        
        # 메모리 구성 요소들
        self._initialize_memory_components()
        
    def _initialize_memory_components(self):
        """메모리 구성 요소들을 초기화합니다."""
        
        # 메모리 슬롯들
        self.memory_slots = []
        self.access_count = 0
        
        # 현재 상태 설정
        self.current_state = {
            'memory_slots': self.memory_slots,
            'utilization': 0.0,
            'access_count': self.access_count,
            'timestamp': datetime.now().isoformat()
        }
        
    def store(self, content: torch.Tensor, priority: float = 1.0, 
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        메모리에 내용 저장
        
        Args:
            content: 저장할 내용
            priority: 우선순위
            metadata: 메타데이터
            
        Returns:
            저장된 항목의 ID
        """
        # 메모리 공간 확보
        if self.current_state['utilization'] >= 1.0:
            self._evict_low_priority_items()
        
        # 새 메모리 항목 생성
        item_id = f"mem_{len(self.memory_slots)}_{datetime.now().timestamp()}"
        memory_item = {
            'id': item_id,
            'content': content,
            'priority': priority,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'access_count': 0,
            'last_access': datetime.now().isoformat(),
            'decay_factor': 1.0
        }
        
        self.memory_slots.append(memory_item)
        self.current_state['utilization'] = len(self.memory_slots) / self.capacity
        
        return item_id
    
    def retrieve(self, item_id: str) -> Optional[torch.Tensor]:
        """
        메모리에서 항목 검색
        
        Args:
            item_id: 항목 ID
            
        Returns:
            검색된 내용 (없으면 None)
        """
        for item in self.memory_slots:
            if item['id'] == item_id:
                # 접근 기록 업데이트
                item['access_count'] += 1
                item['last_access'] = datetime.now().isoformat()
                item['decay_factor'] = min(1.0, item['decay_factor'] + 0.1)
                
                self.access_count += 1
                self.current_state['access_count'] = self.access_count
                
                return item['content']
        
        return None
    
    def _evict_low_priority_items(self):
        """낮은 우선순위 항목들을 제거합니다."""
        if not self.memory_slots:
            return
        
        # 우선순위와 접근 빈도를 고려한 점수 계산
        for item in self.memory_slots:
            time_diff = (datetime.now() - datetime.fromisoformat(item['last_access'])).total_seconds()
            decay = np.exp(-self.decay_rate * time_diff / 3600)  # 시간당 감쇠
            item['decay_factor'] *= decay
            
            # 종합 점수 계산
            item['score'] = (
                item['priority'] * 
                item['decay_factor'] * 
                (1 + item['access_count'] * 0.1)
            )
        
        # 점수 순으로 정렬하여 낮은 점수 항목 제거
        self.memory_slots.sort(key=lambda x: x['score'], reverse=True)
        
        # 용량의 20%를 유지
        keep_count = max(1, int(self.capacity * 0.2))
        self.memory_slots = self.memory_slots[:keep_count]
        
        self.current_state['utilization'] = len(self.memory_slots) / self.capacity
    
    def consolidate(self) -> List[str]:
        """
        메모리 통합 수행
        
        Returns:
            통합된 항목 ID 목록
        """
        consolidated_items = []
        
        # 유사한 항목들을 그룹화
        groups = self._group_similar_items()
        
        for group in groups:
            if len(group) > 1:
                # 그룹 통합
                consolidated_item = self._merge_items(group)
                consolidated_items.append(consolidated_item['id'])
                
                # 원본 항목들 제거
                for item in group:
                    self.memory_slots.remove(item)
                
                # 통합된 항목 추가
                self.memory_slots.append(consolidated_item)
        
        self.current_state['utilization'] = len(self.memory_slots) / self.capacity
        return consolidated_items
    
    def _group_similar_items(self) -> List[List[Dict[str, Any]]]:
        """유사한 항목들을 그룹화합니다."""
        groups = []
        used_indices = set()
        
        for i, item1 in enumerate(self.memory_slots):
            if i in used_indices:
                continue
                
            group = [item1]
            used_indices.add(i)
            
            for j, item2 in enumerate(self.memory_slots[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # 유사도 계산
                similarity = self._calculate_similarity(item1, item2)
                if similarity > self.consolidation_threshold:
                    group.append(item2)
                    used_indices.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, item1: Dict[str, Any], 
                            item2: Dict[str, Any]) -> float:
        """두 항목 간의 유사도를 계산합니다."""
        # 내용 유사도
        content_sim = torch.cosine_similarity(
            item1['content'].flatten().unsqueeze(0),
            item2['content'].flatten().unsqueeze(0)
        ).item()
        
        # 메타데이터 유사도
        metadata_sim = 0.0
        if item1['metadata'] and item2['metadata']:
            common_keys = set(item1['metadata'].keys()) & set(item2['metadata'].keys())
            if common_keys:
                metadata_sim = len(common_keys) / max(len(item1['metadata']), len(item2['metadata']))
        
        return (content_sim + metadata_sim) / 2
    
    def _merge_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여러 항목을 하나로 통합합니다."""
        # 내용 통합 (가중 평균)
        total_weight = sum(item['priority'] for item in items)
        merged_content = torch.zeros_like(items[0]['content'])
        
        for item in items:
            weight = item['priority'] / total_weight
            merged_content += weight * item['content']
        
        # 메타데이터 통합
        merged_metadata = {}
        for item in items:
            merged_metadata.update(item['metadata'])
        
        # 통합된 항목 생성
        merged_item = {
            'id': f"consolidated_{datetime.now().timestamp()}",
            'content': merged_content,
            'priority': sum(item['priority'] for item in items) / len(items),
            'metadata': merged_metadata,
            'timestamp': datetime.now().isoformat(),
            'access_count': sum(item['access_count'] for item in items),
            'last_access': datetime.now().isoformat(),
            'decay_factor': 1.0
        }
        
        return merged_item
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        메모리 통계 반환
        
        Returns:
            메모리 통계 딕셔너리
        """
        if not self.memory_slots:
            return {
                'total_items': 0,
                'utilization': 0.0,
                'avg_priority': 0.0,
                'avg_access_count': 0.0
            }
        
        priorities = [item['priority'] for item in self.memory_slots]
        access_counts = [item['access_count'] for item in self.memory_slots]
        
        return {
            'total_items': len(self.memory_slots),
            'utilization': self.current_state['utilization'],
            'avg_priority': np.mean(priorities),
            'avg_access_count': np.mean(access_counts),
            'max_priority': max(priorities),
            'min_priority': min(priorities)
        }
    
    def search(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        메모리에서 유사한 항목 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 항목 수
            
        Returns:
            (항목 ID, 유사도) 튜플 목록
        """
        similarities = []
        
        for item in self.memory_slots:
            similarity = torch.cosine_similarity(
                query.flatten().unsqueeze(0),
                item['content'].flatten().unsqueeze(0)
            ).item()
            
            # 우선순위와 접근 빈도를 고려한 가중 유사도
            weighted_similarity = similarity * item['priority'] * (1 + item['access_count'] * 0.1)
            similarities.append((item['id'], weighted_similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear(self):
        """메모리 초기화"""
        self.memory_slots = []
        self.access_count = 0
        self.current_state['utilization'] = 0.0
        self.current_state['access_count'] = 0
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """상태 업데이트"""
        self.current_state.update(new_state)
        self.current_state['timestamp'] = datetime.now().isoformat()
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.current_state.copy()
    
    def get_state_at_time(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """특정 시간의 상태 반환 (기본 구현)"""
        # 간단한 구현: 현재 상태 반환
        return self.current_state.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            'name': getattr(self, 'name', 'WorkingMemory'),
            'module_type': 'working_memory'
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """모듈 복잡도를 반환합니다."""
        return {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'module_type': 'working_memory'
        }
    
    def reset(self) -> None:
        """작업 메모리 초기화"""
        super().reset()
        self._initialize_memory_components()
