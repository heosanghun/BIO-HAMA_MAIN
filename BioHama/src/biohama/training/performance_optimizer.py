"""
성능 최적화기 (Performance Optimizer)

선호도 모델과 보상 계산기의 성능을 최적화하는 시스템입니다.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import gc
from collections import deque

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """메모리 최적화기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_threshold = config.get('memory_threshold', 0.8)
        self.compression_ratio = config.get('compression_ratio', 0.5)
        self.garbage_collection_freq = config.get('garbage_collection_freq', 100)
        
        # 메모리 사용량 추적
        self.memory_usage_history = deque(maxlen=1000)
        self.optimization_count = 0
        
    def optimize_memory_usage(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        optimization_results = {}
        
        # 현재 메모리 사용량 확인
        current_memory = self._get_memory_usage()
        self.memory_usage_history.append(current_memory)
        
        # 메모리 사용량이 임계값을 초과하면 최적화 수행
        if current_memory > self.memory_threshold:
            logger.info(f"메모리 최적화 시작 (사용량: {current_memory:.2f}%)")
            
            # 가비지 컬렉션
            self._perform_garbage_collection()
            
            # 모델별 메모리 최적화
            for name, model in models.items():
                if hasattr(model, 'memory'):
                    # 선호도 메모리 최적화
                    model.memory.optimize_memory()
                    optimization_results[name] = 'memory_optimized'
                    
                if hasattr(model, 'learner'):
                    # 학습기 히스토리 정리
                    if len(model.learner.learning_history) > 500:
                        # 오래된 히스토리 제거
                        model.learner.learning_history.clear()
                        optimization_results[name] = 'learner_history_cleared'
                        
            # 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.optimization_count += 1
            
        return optimization_results
        
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 확인"""
        if torch.cuda.is_available():
            # GPU 메모리 사용량
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) * 100
        else:
            # CPU 메모리 사용량 (간단한 추정)
            import psutil
            return psutil.virtual_memory().percent
            
    def _perform_garbage_collection(self):
        """가비지 컬렉션 수행"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        current_memory = self._get_memory_usage()
        
        if self.memory_usage_history:
            avg_memory = np.mean(list(self.memory_usage_history))
            max_memory = np.max(list(self.memory_usage_history))
        else:
            avg_memory = current_memory
            max_memory = current_memory
            
        return {
            'current_memory_usage': current_memory,
            'average_memory_usage': avg_memory,
            'max_memory_usage': max_memory,
            'optimization_count': self.optimization_count,
            'memory_threshold': self.memory_threshold
        }


class ComputationOptimizer:
    """계산 최적화기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size_optimization = config.get('batch_size_optimization', True)
        self.mixed_precision = config.get('mixed_precision', True)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # 성능 추적
        self.computation_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=1000)
        
    def optimize_computation(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """계산 최적화"""
        optimization_results = {}
        
        for name, model in models.items():
            # 혼합 정밀도 적용
            if self.mixed_precision and hasattr(model, 'to'):
                if torch.cuda.is_available():
                    model.half()  # FP16으로 변환
                    optimization_results[name] = 'mixed_precision_applied'
                    
            # 그래디언트 체크포인팅 적용
            if self.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                optimization_results[name] = 'gradient_checkpointing_enabled'
                
        return optimization_results
        
    def measure_computation_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """계산 시간 측정"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        computation_time = end_time - start_time
        self.computation_times.append(computation_time)
        
        return result, computation_time
        
    def optimize_batch_size(self, current_batch_size: int, memory_usage: float) -> int:
        """배치 크기 최적화"""
        if not self.batch_size_optimization:
            return current_batch_size
            
        if memory_usage > 0.9:  # 메모리 사용량이 90% 이상
            return max(1, current_batch_size // 2)
        elif memory_usage < 0.5:  # 메모리 사용량이 50% 미만
            return current_batch_size * 2
        else:
            return current_batch_size
            
    def get_computation_stats(self) -> Dict[str, Any]:
        """계산 통계 반환"""
        if not self.computation_times:
            return {}
            
        times = list(self.computation_times)
        return {
            'avg_computation_time': np.mean(times),
            'min_computation_time': np.min(times),
            'max_computation_time': np.max(times),
            'std_computation_time': np.std(times),
            'total_computations': len(times)
        }


class RealTimeOptimizer:
    """실시간 처리 최적화기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_latency = config.get('target_latency', 0.1)  # 100ms
        self.adaptive_processing = config.get('adaptive_processing', True)
        
        # 실시간 성능 추적
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=1000)
        
    def optimize_for_realtime(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 처리를 위한 최적화"""
        optimization_results = {}
        
        # 현재 지연시간 측정
        current_latency = self._measure_current_latency(models)
        self.latency_history.append(current_latency)
        
        # 지연시간이 목표를 초과하면 최적화 수행
        if current_latency > self.target_latency:
            logger.info(f"실시간 최적화 시작 (지연시간: {current_latency:.3f}s)")
            
            # 적응형 처리 적용
            if self.adaptive_processing:
                for name, model in models.items():
                    if hasattr(model, 'memory'):
                        # 메모리 크기 줄이기
                        if hasattr(model.memory, 'max_size'):
                            model.memory.max_size = max(100, model.memory.max_size // 2)
                            optimization_results[name] = 'memory_size_reduced'
                            
                    if hasattr(model, 'learner'):
                        # 학습 히스토리 크기 줄이기
                        if hasattr(model.learner, 'memory_size'):
                            model.learner.memory_size = max(100, model.learner.memory_size // 2)
                            optimization_results[name] = 'learner_memory_reduced'
                            
        return optimization_results
        
    def _measure_current_latency(self, models: Dict[str, Any]) -> float:
        """현재 지연시간 측정"""
        start_time = time.time()
        
        # 간단한 추론 작업 수행
        for name, model in models.items():
            if hasattr(model, 'forward'):
                # 더미 입력으로 지연시간 측정
                dummy_input = torch.randn(1, 512)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    
                with torch.no_grad():
                    _ = model(dummy_input)
                    
        end_time = time.time()
        return end_time - start_time
        
    def get_realtime_stats(self) -> Dict[str, Any]:
        """실시간 처리 통계 반환"""
        if not self.latency_history:
            return {}
            
        latencies = list(self.latency_history)
        return {
            'current_latency': latencies[-1] if latencies else 0.0,
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'target_latency': self.target_latency,
            'latency_violations': sum(1 for l in latencies if l > self.target_latency)
        }


class PerformanceOptimizer:
    """성능 최적화기 메인 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 하위 최적화기들
        self.memory_optimizer = MemoryOptimizer(
            config.get('memory_optimizer', {})
        )
        
        self.computation_optimizer = ComputationOptimizer(
            config.get('computation_optimizer', {})
        )
        
        self.realtime_optimizer = RealTimeOptimizer(
            config.get('realtime_optimizer', {})
        )
        
        # 최적화 통계
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_optimizations': 0,
            'computation_optimizations': 0,
            'realtime_optimizations': 0
        }
        
        logger.info("성능 최적화기가 초기화되었습니다.")
        
    def optimize_system(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """전체 시스템 최적화"""
        optimization_results = {}
        
        # 메모리 최적화
        memory_results = self.memory_optimizer.optimize_memory_usage(models)
        if memory_results:
            optimization_results['memory'] = memory_results
            self.optimization_stats['memory_optimizations'] += 1
            
        # 계산 최적화
        computation_results = self.computation_optimizer.optimize_computation(models)
        if computation_results:
            optimization_results['computation'] = computation_results
            self.optimization_stats['computation_optimizations'] += 1
            
        # 실시간 최적화
        realtime_results = self.realtime_optimizer.optimize_for_realtime(models)
        if realtime_results:
            optimization_results['realtime'] = realtime_results
            self.optimization_stats['realtime_optimizations'] += 1
            
        if optimization_results:
            self.optimization_stats['total_optimizations'] += 1
            logger.info(f"시스템 최적화 완료: {optimization_results}")
            
        return optimization_results
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            'optimization_stats': self.optimization_stats.copy(),
            'memory_stats': self.memory_optimizer.get_memory_stats(),
            'computation_stats': self.computation_optimizer.get_computation_stats(),
            'realtime_stats': self.realtime_optimizer.get_realtime_stats()
        }
        
    def reset_stats(self):
        """통계 초기화"""
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_optimizations': 0,
            'computation_optimizations': 0,
            'realtime_optimizations': 0
        }
        
        # 하위 최적화기 통계도 초기화
        self.memory_optimizer.memory_usage_history.clear()
        self.computation_optimizer.computation_times.clear()
        self.realtime_optimizer.latency_history.clear()
