"""
프로파일링 모듈

BioHama 시스템의 성능 분석 및 최적화를 제공합니다.
"""

import time
import psutil
import torch
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
import functools


class Profiler:
    """성능 프로파일러"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_usage = []
        
    def start_timer(self, name: str) -> None:
        """타이머 시작"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 시간 반환"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0.0
    
    def record_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 기록"""
        memory_info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_memory_used_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        self.memory_usage.append(memory_info)
        return memory_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {}
        
        for name, times in self.metrics.items():
            if times:
                stats[name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        if self.memory_usage:
            stats['memory'] = {
                'avg_cpu_percent': sum(m['cpu_percent'] for m in self.memory_usage) / len(self.memory_usage),
                'avg_memory_percent': sum(m['memory_percent'] for m in self.memory_usage) / len(self.memory_usage),
                'avg_memory_used_mb': sum(m['memory_used_mb'] for m in self.memory_usage) / len(self.memory_usage)
            }
        
        return stats
    
    def reset(self) -> None:
        """프로파일러 초기화"""
        self.metrics.clear()
        self.start_times.clear()
        self.memory_usage.clear()


# 전역 프로파일러 인스턴스
_global_profiler = Profiler()


@contextmanager
def performance_monitor(name: str):
    """성능 모니터링 컨텍스트 매니저"""
    _global_profiler.start_timer(name)
    try:
        yield
    finally:
        _global_profiler.end_timer(name)


def profile_function(func: Callable) -> Callable:
    """함수 프로파일링 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with performance_monitor(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def get_profiler() -> Profiler:
    """전역 프로파일러 반환"""
    return _global_profiler

