"""
메모리 관리 모듈

BioHama 시스템의 효율적인 메모리 사용을 관리합니다.
"""

import torch
import gc
import psutil
from typing import Any, Dict, List, Optional
import numpy as np


class MemoryManager:
    """메모리 관리자"""
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        메모리 관리자 초기화
        
        Args:
            max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self.max_memory_mb = max_memory_mb
        self.memory_history = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        memory_info = {
            'cpu_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_used_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_free_mb': (torch.cuda.get_device_properties(0).total_memory - 
                                     torch.cuda.memory_reserved()) / 1024 / 1024
            })
        
        self.memory_history.append(memory_info)
        
        # 히스토리 크기 제한
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
        
        return memory_info
    
    def is_memory_high(self, threshold_percent: float = 80.0) -> bool:
        """메모리 사용량이 높은지 확인"""
        memory_info = self.get_memory_usage()
        return memory_info['cpu_memory_percent'] > threshold_percent
    
    def clear_memory(self) -> Dict[str, float]:
        """메모리 정리"""
        # 가비지 컬렉션
        gc.collect()
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 정리 후 메모리 사용량 반환
        return self.get_memory_usage()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        before_memory = self.get_memory_usage()
        
        # 메모리 정리
        after_memory = self.clear_memory()
        
        # 최적화 결과
        optimization_result = {
            'before': before_memory,
            'after': after_memory,
            'cpu_memory_freed_mb': before_memory['cpu_memory_mb'] - after_memory['cpu_memory_mb']
        }
        
        if torch.cuda.is_available():
            optimization_result['gpu_memory_freed_mb'] = (
                before_memory['gpu_memory_used_mb'] - after_memory['gpu_memory_used_mb']
            )
        
        return optimization_result
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        if not self.memory_history:
            return {}
        
        cpu_memory_usage = [m['cpu_memory_mb'] for m in self.memory_history]
        gpu_memory_usage = [m.get('gpu_memory_used_mb', 0) for m in self.memory_history]
        
        stats = {
            'cpu_memory': {
                'avg_mb': np.mean(cpu_memory_usage),
                'max_mb': np.max(cpu_memory_usage),
                'min_mb': np.min(cpu_memory_usage),
                'std_mb': np.std(cpu_memory_usage)
            },
            'gpu_memory': {
                'avg_mb': np.mean(gpu_memory_usage),
                'max_mb': np.max(gpu_memory_usage),
                'min_mb': np.min(gpu_memory_usage),
                'std_mb': np.std(gpu_memory_usage)
            },
            'history_length': len(self.memory_history)
        }
        
        return stats
    
    def monitor_memory_usage(self, callback: Optional[callable] = None) -> None:
        """메모리 사용량 모니터링"""
        memory_info = self.get_memory_usage()
        
        if callback:
            callback(memory_info)
        
        # 메모리 사용량이 높으면 자동 정리
        if self.is_memory_high():
            self.clear_memory()
    
    def reset(self) -> None:
        """메모리 관리자 초기화"""
        self.memory_history.clear()
        self.clear_memory()

