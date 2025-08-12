"""
연산 종료 모듈 (M_terminate)

이 모듈은 신뢰도 추정과 조기 종료 로직을 통해 효율적인 처리 종료를 제공합니다.

주요 기능:
- 신뢰도 기반 종료 판단
- 조기 종료 메커니즘
- 처리 품질 평가
- 동적 종료 조건 조정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import logging
from datetime import datetime

from .base.module_interface import ModuleInterface
from .base.state_interface import StateInterface

logger = logging.getLogger(__name__)


class ConfidenceEstimator(nn.Module):
    """신뢰도 추정기"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_confidence_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_confidence_heads = num_confidence_heads
        
        # 신뢰도 추정 네트워크
        self.confidence_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 다중 신뢰도 헤드
        self.confidence_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_confidence_heads)
        ])
        
        # 신뢰도 통합 레이어
        self.confidence_integrator = nn.Sequential(
            nn.Linear(num_confidence_heads, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        신뢰도 추정
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)
            
        Returns:
            confidence: 통합 신뢰도 점수
            head_confidences: 개별 헤드 신뢰도 점수들
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 인코딩
        encoded = self.confidence_encoder(x)  # (batch_size, seq_len, hidden_dim//2)
        
        # 개별 헤드 신뢰도 계산
        head_confidences = []
        for head in self.confidence_heads:
            head_conf = head(encoded)  # (batch_size, seq_len, 1)
            head_confidences.append(head_conf)
        
        # 헤드 신뢰도 결합
        head_confidences = torch.cat(head_confidences, dim=-1)  # (batch_size, seq_len, num_heads)
        
        # 통합 신뢰도 계산
        confidence = self.confidence_integrator(head_confidences)  # (batch_size, seq_len, 1)
        
        return confidence.squeeze(-1), head_confidences
    
    def get_confidence_statistics(self, confidence: torch.Tensor) -> Dict[str, float]:
        """신뢰도 통계 계산"""
        return {
            'mean_confidence': confidence.mean().item(),
            'std_confidence': confidence.std().item(),
            'min_confidence': confidence.min().item(),
            'max_confidence': confidence.max().item(),
            'high_confidence_ratio': (confidence > 0.8).float().mean().item()
        }


class QualityEvaluator(nn.Module):
    """품질 평가기"""
    
    def __init__(self,
                 input_dim: int,
                 quality_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.quality_dim = quality_dim
        
        # 품질 평가 네트워크
        self.quality_encoder = nn.Sequential(
            nn.Linear(input_dim, quality_dim * 2),
            nn.ReLU(),
            nn.Linear(quality_dim * 2, quality_dim),
            nn.ReLU(),
            nn.Linear(quality_dim, 5)  # 5가지 품질 지표
        )
        
        # 품질 지표 가중치
        self.quality_weights = nn.Parameter(torch.ones(5) / 5)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        품질 평가
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)
            
        Returns:
            overall_quality: 전체 품질 점수
            quality_indicators: 개별 품질 지표들
        """
        batch_size, seq_len, _ = x.shape
        
        # 품질 지표 계산
        quality_indicators = self.quality_encoder(x)  # (batch_size, seq_len, 5)
        
        # 가중치 적용
        weighted_quality = quality_indicators * self.quality_weights.unsqueeze(0).unsqueeze(0)
        
        # 전체 품질 점수
        overall_quality = weighted_quality.sum(dim=-1)  # (batch_size, seq_len)
        
        return overall_quality, quality_indicators
    
    def get_quality_breakdown(self, quality_indicators: torch.Tensor) -> Dict[str, torch.Tensor]:
        """품질 분석"""
        quality_names = ['coherence', 'completeness', 'accuracy', 'relevance', 'consistency']
        
        breakdown = {}
        for i, name in enumerate(quality_names):
            breakdown[name] = quality_indicators[:, :, i]
            
        return breakdown


class EarlyStopping(nn.Module):
    """조기 종료 모듈"""
    
    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 0.001,
                 min_confidence: float = 0.7,
                 max_iterations: int = 100):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.min_confidence = min_confidence
        self.max_iterations = max_iterations
        
        # 종료 조건 추적
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False
        
    def forward(self,
                confidence: torch.Tensor,
                quality: torch.Tensor,
                iteration: int) -> Dict[str, Any]:
        """
        조기 종료 판단
        
        Args:
            confidence: 신뢰도 점수
            quality: 품질 점수
            iteration: 현재 반복 횟수
            
        Returns:
            stop_info: 종료 관련 정보
        """
        # 현재 점수 계산 (신뢰도와 품질의 가중 평균)
        current_score = (confidence.mean() + quality.mean()) / 2
        
        # 종료 조건 체크
        stop_reasons = []
        
        # 1. 최대 반복 횟수 도달
        if iteration >= self.max_iterations:
            stop_reasons.append('max_iterations')
            
        # 2. 최소 신뢰도 달성
        if confidence.mean() >= self.min_confidence:
            stop_reasons.append('confidence_threshold')
            
        # 3. 개선 없음 (patience 기반)
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            stop_reasons.append('no_improvement')
            
        # 종료 결정
        should_stop = len(stop_reasons) > 0
        
        return {
            'should_stop': should_stop,
            'stop_reasons': stop_reasons,
            'current_score': current_score.item(),
            'best_score': self.best_score,
            'counter': self.counter,
            'iteration': iteration
        }
    
    def reset(self):
        """상태 초기화"""
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False


class TerminationModule(ModuleInterface, nn.Module):
    """연산 종료 모듈"""
    
    def __init__(self,
                 config: Dict[str, Any]):
        ModuleInterface.__init__(self, config)
        nn.Module.__init__(self)
        
        # 설정 파라미터
        self.input_dim = config.get('input_dim', 512)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.quality_threshold = config.get('quality_threshold', 0.6)
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 0.001)
        self.max_iterations = config.get('max_iterations', 100)
        
        # 모듈 구성 요소
        self.confidence_estimator = ConfidenceEstimator(
            input_dim=self.input_dim
        )
        
        self.quality_evaluator = QualityEvaluator(
            input_dim=self.input_dim
        )
        
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            min_confidence=self.confidence_threshold,
            max_iterations=self.max_iterations
        )
        
        # 통계 추적
        self.termination_stats = {
            'total_checks': 0,
            'early_terminations': 0,
            'avg_iterations': 0.0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0
        }
        
        # 활성화 상태
        self.is_active = True
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        종료 판단 처리
        
        Args:
            inputs: 입력 데이터
                - 'features': 특징 텐서
                - 'iteration': 현재 반복 횟수
                - 'force_continue': 강제 계속 플래그
                
        Returns:
            outputs: 처리 결과
                - 'should_terminate': 종료 여부
                - 'confidence': 신뢰도 점수
                - 'quality': 품질 점수
                - 'stop_reasons': 종료 이유들
                - 'termination_stats': 종료 통계
        """
        if not self.is_active:
            return inputs
            
        features = inputs.get('features')
        iteration = inputs.get('iteration', 0)
        force_continue = inputs.get('force_continue', False)
        
        if features is None:
            logger.warning("TerminationModule: features가 제공되지 않았습니다.")
            return inputs
            
        # 1. 신뢰도 추정
        confidence, head_confidences = self.confidence_estimator(features)
        
        # 2. 품질 평가
        quality, quality_indicators = self.quality_evaluator(features)
        
        # 3. 조기 종료 판단
        stop_info = self.early_stopping(confidence, quality, iteration)
        
        # 4. 강제 계속 처리
        if force_continue:
            stop_info['should_stop'] = False
            stop_info['stop_reasons'] = []
        
        # 5. 통계 업데이트
        self._update_stats(stop_info, confidence, quality, iteration)
        
        # 6. 상태 업데이트
        self.update_state({
            'last_confidence': confidence.detach(),
            'last_quality': quality.detach(),
            'last_iteration': iteration,
            'last_should_terminate': stop_info['should_stop']
        })
        
        return {
            'should_terminate': stop_info['should_stop'],
            'confidence': confidence,
            'quality': quality,
            'stop_reasons': stop_info['stop_reasons'],
            'iteration': iteration,
            'confidence_stats': self.confidence_estimator.get_confidence_statistics(confidence),
            'quality_breakdown': self.quality_evaluator.get_quality_breakdown(quality_indicators),
            'termination_stats': self.termination_stats.copy()
        }
    
    def _update_stats(self,
                     stop_info: Dict[str, Any],
                     confidence: torch.Tensor,
                     quality: torch.Tensor,
                     iteration: int):
        """통계 업데이트"""
        self.termination_stats['total_checks'] += 1
        
        if stop_info['should_stop']:
            self.termination_stats['early_terminations'] += 1
            
        # 평균 계산
        total_checks = self.termination_stats['total_checks']
        current_avg_iterations = self.termination_stats['avg_iterations']
        current_avg_confidence = self.termination_stats['avg_confidence']
        current_avg_quality = self.termination_stats['avg_quality']
        
        self.termination_stats['avg_iterations'] = (
            (current_avg_iterations * (total_checks - 1) + iteration) / total_checks
        )
        self.termination_stats['avg_confidence'] = (
            (current_avg_confidence * (total_checks - 1) + confidence.mean().item()) / total_checks
        )
        self.termination_stats['avg_quality'] = (
            (current_avg_quality * (total_checks - 1) + quality.mean().item()) / total_checks
        )
    
    def update_state(self, new_state: Dict[str, Any]):
        """상태 업데이트"""
        if not hasattr(self, 'state'):
            self.state = {}
        self.state.update(new_state)
    
    def get_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return getattr(self, 'state', {})
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터 반환"""
        return {
            'name': self.name,
            'module_type': 'termination',
            'confidence_threshold': self.confidence_threshold,
            'quality_threshold': self.quality_threshold,
            'patience': self.patience,
            'is_active': self.is_active
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """복잡도 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': 'termination'
        }
    
    def activate(self):
        """모듈 활성화"""
        self.is_active = True
        logger.info(f"{self.name} 모듈이 활성화되었습니다.")
    
    def deactivate(self):
        """모듈 비활성화"""
        self.is_active = False
        logger.info(f"{self.name} 모듈이 비활성화되었습니다.")
    
    def reset_early_stopping(self):
        """조기 종료 상태 초기화"""
        self.early_stopping.reset()
        logger.info(f"{self.name}: 조기 종료 상태가 초기화되었습니다.")
    
    def set_thresholds(self, confidence_threshold: float = None, quality_threshold: float = None):
        """임계값 설정"""
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
            
        if quality_threshold is not None:
            self.quality_threshold = max(0.0, min(1.0, quality_threshold))
            
        logger.info(f"{self.name}: 임계값이 업데이트되었습니다. "
                   f"신뢰도: {self.confidence_threshold:.2f}, 품질: {self.quality_threshold:.2f}")
    
    def get_termination_stats(self) -> Dict[str, Any]:
        """종료 통계 반환"""
        return self.termination_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.termination_stats = {
            'total_checks': 0,
            'early_terminations': 0,
            'avg_iterations': 0.0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0
        }


class TerminationState(StateInterface):
    """종료 상태 관리"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.termination_history = []
        self.confidence_history = []
        self.quality_history = []
        self.performance_metrics = {}
        
    def update_state(self, new_state: Dict[str, Any]):
        """상태 업데이트"""
        timestamp = len(self.termination_history)
        
        self.termination_history.append({
            'timestamp': timestamp,
            'should_terminate': new_state.get('should_terminate', False),
            'stop_reasons': new_state.get('stop_reasons', []),
            'iteration': new_state.get('iteration', 0),
            'confidence': new_state.get('confidence'),
            'quality': new_state.get('quality')
        })
        
        # 성능 메트릭 업데이트
        if 'termination_stats' in new_state:
            self.performance_metrics.update(new_state['termination_stats'])
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        if not self.termination_history:
            return {}
            
        latest = self.termination_history[-1]
        return {
            'current_should_terminate': latest['should_terminate'],
            'current_stop_reasons': latest['stop_reasons'],
            'current_iteration': latest['iteration'],
            'current_confidence': latest['confidence'],
            'current_quality': latest['quality'],
            'performance_metrics': self.performance_metrics
        }
    
    def get_state_at_time(self, timestamp: int) -> Dict[str, Any]:
        """특정 시점의 상태 반환"""
        if 0 <= timestamp < len(self.termination_history):
            return self.termination_history[timestamp]
        return {}
    
    def get_termination_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """종료 트렌드 분석"""
        if len(self.termination_history) < window_size:
            return {}
            
        recent_history = self.termination_history[-window_size:]
        
        termination_rates = [1.0 if h['should_terminate'] else 0.0 for h in recent_history]
        iterations = [h['iteration'] for h in recent_history]
        
        return {
            'avg_termination_rate': np.mean(termination_rates),
            'avg_iterations': np.mean(iterations),
            'termination_trend': np.polyfit(range(len(termination_rates)), termination_rates, 1)[0],
            'iteration_trend': np.polyfit(range(len(iterations)), iterations, 1)[0]
        }
