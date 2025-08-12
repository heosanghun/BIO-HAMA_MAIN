"""
선호도 메모리 (Preference Memory)

선호도 데이터를 효율적으로 저장하고 검색하는 메모리 시스템입니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import deque, OrderedDict
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreferenceData:
    """선호도 데이터 구조"""
    preference_type: str  # 'explicit', 'implicit', 'relative', 'absolute'
    input_data: torch.Tensor
    preference_value: Union[float, int, torch.Tensor]
    timestamp: float
    context: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PreferenceMemory:
    """선호도 메모리"""
    
    def __init__(self,
                 max_size: int = 10000,
                 memory_type: str = 'fifo',
                 similarity_threshold: float = 0.8):
        self.max_size = max_size
        self.memory_type = memory_type
        self.similarity_threshold = similarity_threshold
        
        # 메모리 저장소 초기화
        if memory_type == 'fifo':
            self.memory = deque(maxlen=max_size)
        elif memory_type == 'lru':
            self.memory = OrderedDict()
        else:
            raise ValueError(f"지원하지 않는 메모리 타입: {memory_type}")
            
        # 메모리 인덱스
        self.memory_index = {}
        self.type_index = {}  # 타입별 인덱스
        
        # 메모리 통계
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'similarity_searches': 0,
            'avg_similarity_score': 0.0
        }
        
        # 메모리 최적화 파라미터
        self.compression_enabled = True
        self.compression_ratio = 0.5
        
    def store(self, 
              preference_data: PreferenceData,
              key: Optional[str] = None) -> str:
        """
        선호도 데이터 저장
        
        Args:
            preference_data: 저장할 선호도 데이터
            key: 데이터 키 (None이면 자동 생성)
            
        Returns:
            key: 저장된 데이터의 키
        """
        if key is None:
            key = f"pref_{len(self.memory)}_{int(time.time())}"
            
        # 데이터 압축 (선택사항)
        if self.compression_enabled:
            compressed_data = self._compress_data(preference_data)
        else:
            compressed_data = preference_data
            
        # 메모리에 저장
        if self.memory_type == 'fifo':
            self.memory.append((key, compressed_data))
        elif self.memory_type == 'lru':
            self.memory[key] = compressed_data
            # 크기 제한 확인
            if len(self.memory) > self.max_size:
                oldest_key = next(iter(self.memory))
                del self.memory[oldest_key]
                if oldest_key in self.memory_index:
                    del self.memory_index[oldest_key]
                    
        # 인덱스 업데이트
        self.memory_index[key] = len(self.memory) - 1
        
        # 타입별 인덱스 업데이트
        pref_type = preference_data.preference_type
        if pref_type not in self.type_index:
            self.type_index[pref_type] = []
        self.type_index[pref_type].append(key)
        
        self.stats['total_stored'] += 1
        
        return key
        
    def retrieve(self, key: str) -> Optional[PreferenceData]:
        """
        선호도 데이터 검색
        
        Args:
            key: 검색할 데이터 키
            
        Returns:
            preference_data: 검색된 선호도 데이터
        """
        if key in self.memory_index:
            if self.memory_type == 'fifo':
                idx = self.memory_index[key]
                if idx < len(self.memory):
                    self.stats['memory_hits'] += 1
                    self.stats['total_retrieved'] += 1
                    return self.memory[idx][1]
            elif self.memory_type == 'lru':
                if key in self.memory:
                    # LRU 업데이트
                    data = self.memory.pop(key)
                    self.memory[key] = data
                    self.stats['memory_hits'] += 1
                    self.stats['total_retrieved'] += 1
                    return data
                    
        self.stats['memory_misses'] += 1
        return None
        
    def retrieve_similar(self,
                        query_data: torch.Tensor,
                        top_k: int = 5,
                        preference_type: Optional[str] = None) -> List[Tuple[str, PreferenceData, float]]:
        """
        유사한 선호도 데이터 검색
        
        Args:
            query_data: 쿼리 데이터
            top_k: 반환할 최대 개수
            preference_type: 선호도 타입 필터
            
        Returns:
            similar_data: 유사한 데이터 리스트 (키, 데이터, 유사도)
        """
        if not self.memory:
            return []
            
        similarities = []
        search_keys = []
        
        # 검색 범위 결정
        if preference_type and preference_type in self.type_index:
            search_keys = self.type_index[preference_type]
        else:
            if self.memory_type == 'fifo':
                search_keys = [item[0] for item in self.memory]
            elif self.memory_type == 'lru':
                search_keys = list(self.memory.keys())
                
        # 유사도 계산
        for key in search_keys:
            data = self.retrieve(key)
            if data is not None:
                similarity = self._calculate_similarity(query_data, data.input_data)
                if similarity >= self.similarity_threshold:
                    similarities.append((key, data, similarity))
                    
        # 유사도 기준으로 정렬
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        self.stats['similarity_searches'] += 1
        if similarities:
            self.stats['avg_similarity_score'] = (
                (self.stats['avg_similarity_score'] * (self.stats['similarity_searches'] - 1) + 
                 np.mean([s[2] for s in similarities])) / self.stats['similarity_searches']
            )
            
        return similarities[:top_k]
        
    def retrieve_by_type(self,
                        preference_type: str,
                        limit: Optional[int] = None) -> List[Tuple[str, PreferenceData]]:
        """
        타입별 선호도 데이터 검색
        
        Args:
            preference_type: 선호도 타입
            limit: 반환할 최대 개수
            
        Returns:
            type_data: 타입별 데이터 리스트
        """
        if preference_type not in self.type_index:
            return []
            
        keys = self.type_index[preference_type]
        if limit:
            keys = keys[-limit:]  # 최근 데이터
            
        results = []
        for key in keys:
            data = self.retrieve(key)
            if data is not None:
                results.append((key, data))
                
        return results
        
    def _calculate_similarity(self,
                            query_data: torch.Tensor,
                            stored_data: torch.Tensor) -> float:
        """유사도 계산"""
        try:
            # 텐서 차원 맞춤
            if query_data.dim() != stored_data.dim():
                # 차원이 다르면 평탄화
                query_flat = query_data.flatten()
                stored_flat = stored_data.flatten()
                
                # 크기 맞춤
                min_size = min(query_flat.size(0), stored_flat.size(0))
                query_flat = query_flat[:min_size]
                stored_flat = stored_flat[:min_size]
            else:
                query_flat = query_data.flatten()
                stored_flat = stored_data.flatten()
                
            # 코사인 유사도 계산
            similarity = F.cosine_similarity(
                query_flat.unsqueeze(0),
                stored_flat.unsqueeze(0)
            ).item()
            
            return max(0.0, similarity)  # 음수 유사도는 0으로 처리
            
        except Exception as e:
            logger.warning(f"유사도 계산 실패: {e}")
            return 0.0
            
    def _compress_data(self, preference_data: PreferenceData) -> PreferenceData:
        """데이터 압축"""
        if not self.compression_enabled:
            return preference_data
            
        # 입력 데이터 압축
        compressed_input = self._compress_tensor(preference_data.input_data)
        
        # 압축된 데이터 생성
        compressed_data = PreferenceData(
            preference_type=preference_data.preference_type,
            input_data=compressed_input,
            preference_value=preference_data.preference_value,
            timestamp=preference_data.timestamp,
            context=preference_data.context,
            confidence=preference_data.confidence,
            metadata={
                **(preference_data.metadata or {}),
                'compressed': True,
                'original_size': preference_data.input_data.numel()
            }
        )
        
        return compressed_data
        
    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """텐서 압축"""
        if tensor.numel() <= 100:  # 작은 텐서는 압축하지 않음
            return tensor
            
        # PCA 기반 압축 (간단한 버전)
        flattened = tensor.flatten()
        target_size = int(flattened.numel() * self.compression_ratio)
        
        if target_size < flattened.numel():
            # 상위 성분만 유지
            sorted_values, _ = torch.sort(flattened.abs(), descending=True)
            threshold = sorted_values[target_size - 1]
            
            compressed = torch.where(
                flattened.abs() >= threshold,
                flattened,
                torch.zeros_like(flattened)
            )
            
            return compressed.view_as(tensor)
        else:
            return tensor
            
    def clear(self):
        """메모리 초기화"""
        if self.memory_type == 'fifo':
            self.memory.clear()
        elif self.memory_type == 'lru':
            self.memory.clear()
            
        self.memory_index.clear()
        self.type_index.clear()
        
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'similarity_searches': 0,
            'avg_similarity_score': 0.0
        }
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        hit_rate = (self.stats['memory_hits'] / 
                   max(1, self.stats['memory_hits'] + self.stats['memory_misses']))
        
        # 타입별 통계
        type_stats = {}
        for pref_type, keys in self.type_index.items():
            type_stats[pref_type] = len(keys)
            
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_size': len(self.memory),
            'memory_type': self.memory_type,
            'type_distribution': type_stats,
            'compression_enabled': self.compression_enabled
        }
        
    def optimize_memory(self):
        """메모리 최적화"""
        if self.memory_type == 'lru':
            # LRU 메모리에서 오래된 항목 제거
            while len(self.memory) > self.max_size * 0.8:  # 80% 수준으로 유지
                oldest_key = next(iter(self.memory))
                del self.memory[oldest_key]
                if oldest_key in self.memory_index:
                    del self.memory_index[oldest_key]
                    
        # 압축 비율 조정
        if self.stats['total_stored'] > 1000:
            self.compression_ratio = max(0.3, self.compression_ratio * 0.95)
            
        logger.info(f"메모리 최적화 완료. 현재 크기: {len(self.memory)}")
        
    def export_memory(self, filepath: str):
        """메모리 내보내기"""
        import pickle
        
        export_data = {
            'memory': list(self.memory) if self.memory_type == 'fifo' else dict(self.memory),
            'memory_index': self.memory_index,
            'type_index': self.type_index,
            'stats': self.stats,
            'config': {
                'max_size': self.max_size,
                'memory_type': self.memory_type,
                'similarity_threshold': self.similarity_threshold
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
            
        logger.info(f"메모리가 {filepath}에 저장되었습니다.")
        
    def import_memory(self, filepath: str):
        """메모리 가져오기"""
        import pickle
        
        with open(filepath, 'rb') as f:
            import_data = pickle.load(f)
            
        self.memory = import_data['memory']
        if self.memory_type == 'fifo':
            self.memory = deque(self.memory, maxlen=self.max_size)
        elif self.memory_type == 'lru':
            self.memory = OrderedDict(self.memory)
            
        self.memory_index = import_data['memory_index']
        self.type_index = import_data['type_index']
        self.stats = import_data['stats']
        
        logger.info(f"메모리가 {filepath}에서 로드되었습니다.")
