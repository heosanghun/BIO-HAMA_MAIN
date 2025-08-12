"""
선호도 학습기 (Preference Learner)

선호도 모델의 학습을 담당하는 시스템으로, 온라인 학습과 배치 학습을 지원합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque

logger = logging.getLogger(__name__)


class PreferenceLearner(nn.Module):
    """선호도 학습기"""
    
    def __init__(self,
                 embedding_dim: int,
                 learning_rate: float = 1e-4,
                 momentum: float = 0.9,
                 memory_size: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.memory_size = memory_size
        
        # 학습 상태 추적
        self.learning_history = deque(maxlen=memory_size)
        self.adaptation_rate = 1.0
        self.learning_mode = 'online'  # 'online' 또는 'batch'
        
        # 온라인 학습 파라미터
        self.online_learning_rate = learning_rate
        self.batch_learning_rate = learning_rate * 0.1
        
        # 적응형 학습 파라미터
        self.min_lr = learning_rate * 0.01
        self.max_lr = learning_rate * 10.0
        self.lr_decay = 0.95
        self.lr_increase = 1.05
        
        # 성능 추적
        self.performance_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_loss': 0.0,
            'avg_improvement': 0.0,
            'convergence_rate': 0.0
        }
        
    def update_preference(self,
                         current_preference: torch.Tensor,
                         target_preference: torch.Tensor,
                         learning_mode: str = 'online',
                         context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        선호도 업데이트
        
        Args:
            current_preference: 현재 선호도
            target_preference: 목표 선호도
            learning_mode: 학습 모드 ('online' 또는 'batch')
            context: 학습 컨텍스트
            
        Returns:
            updated_preference: 업데이트된 선호도
        """
        # 학습률 선택
        lr = self.online_learning_rate if learning_mode == 'online' else self.batch_learning_rate
        
        # 적응형 학습률 적용
        adaptive_lr = self._calculate_adaptive_learning_rate(lr, context)
        
        # 선호도 차이 계산
        preference_diff = target_preference - current_preference
        
        # 업데이트
        updated_preference = current_preference + adaptive_lr * preference_diff
        
        # 학습 히스토리 기록
        self._record_learning_step(
            current_preference, target_preference, preference_diff, adaptive_lr, context
        )
        
        # 적응률 업데이트
        self._update_adaptation_rate(preference_diff)
        
        # 성능 메트릭 업데이트
        self._update_performance_metrics(preference_diff, adaptive_lr)
        
        return updated_preference
        
    def _calculate_adaptive_learning_rate(self,
                                        base_lr: float,
                                        context: Optional[Dict[str, Any]]) -> float:
        """적응형 학습률 계산"""
        lr = base_lr * self.adaptation_rate
        
        # 컨텍스트 기반 조정
        if context is not None:
            # 신뢰도 기반 조정
            if 'confidence' in context:
                confidence = context['confidence']
                lr *= (0.5 + confidence)  # 신뢰도 높을수록 학습률 증가
                
            # 보상 기반 조정
            if 'reward' in context:
                reward = context['reward']
                if reward > 0:
                    lr *= (1.0 + reward * 0.1)  # 긍정적 보상일 때 학습률 증가
                else:
                    lr *= (1.0 - abs(reward) * 0.05)  # 부정적 보상일 때 학습률 감소
                    
            # 시간 기반 조정
            if 'time_since_last_update' in context:
                time_diff = context['time_since_last_update']
                if time_diff > 1.0:  # 1초 이상 지났으면 학습률 증가
                    lr *= (1.0 + min(time_diff * 0.1, 0.5))
                    
        # 학습률 범위 제한
        lr = np.clip(lr, self.min_lr, self.max_lr)
        
        return lr
        
    def _update_adaptation_rate(self, preference_diff: torch.Tensor):
        """적응률 업데이트"""
        # 선호도 변화의 크기에 기반한 적응률 조정
        diff_magnitude = torch.norm(preference_diff).item()
        
        if diff_magnitude > 0.1:
            # 큰 변화: 적응률 증가
            self.adaptation_rate = min(2.0, self.adaptation_rate * self.lr_increase)
        elif diff_magnitude < 0.01:
            # 작은 변화: 적응률 감소
            self.adaptation_rate = max(0.1, self.adaptation_rate * self.lr_decay)
            
    def _record_learning_step(self,
                            current_preference: torch.Tensor,
                            target_preference: torch.Tensor,
                            preference_diff: torch.Tensor,
                            learning_rate: float,
                            context: Optional[Dict[str, Any]]):
        """학습 스텝 기록"""
        step_info = {
            'current': current_preference.detach().cpu().numpy(),
            'target': target_preference.detach().cpu().numpy(),
            'diff': preference_diff.detach().cpu().numpy(),
            'lr': learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'context': context
        }
        
        self.learning_history.append(step_info)
        
    def _update_performance_metrics(self,
                                  preference_diff: torch.Tensor,
                                  learning_rate: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics['total_updates'] += 1
        
        # 성공적인 업데이트 판단
        diff_magnitude = torch.norm(preference_diff).item()
        if diff_magnitude < 0.1:  # 변화가 적당한 크기면 성공
            self.performance_metrics['successful_updates'] += 1
            
        # 평균 손실 업데이트
        loss = diff_magnitude
        n = self.performance_metrics['total_updates']
        self.performance_metrics['avg_loss'] = (
            (self.performance_metrics['avg_loss'] * (n - 1) + loss) / n
        )
        
        # 평균 개선도 업데이트
        improvement = 1.0 / (1.0 + loss)  # 손실이 작을수록 개선도 높음
        self.performance_metrics['avg_improvement'] = (
            (self.performance_metrics['avg_improvement'] * (n - 1) + improvement) / n
        )
        
        # 수렴률 계산
        self.performance_metrics['convergence_rate'] = (
            self.performance_metrics['successful_updates'] / 
            max(1, self.performance_metrics['total_updates'])
        )
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        if not self.learning_history:
            return {}
            
        recent_history = list(self.learning_history)[-100:]
        
        # 기본 통계
        avg_diff = np.mean([np.linalg.norm(h['diff']) for h in recent_history])
        avg_lr = np.mean([h['lr'] for h in recent_history])
        
        # 학습 트렌드 분석
        if len(recent_history) > 10:
            diffs = [np.linalg.norm(h['diff']) for h in recent_history]
            trend = np.polyfit(range(len(diffs)), diffs, 1)[0]
        else:
            trend = 0.0
            
        return {
            'avg_preference_diff': avg_diff,
            'avg_learning_rate': avg_lr,
            'adaptation_rate': self.adaptation_rate,
            'learning_trend': trend,
            'total_updates': self.performance_metrics['total_updates'],
            'convergence_rate': self.performance_metrics['convergence_rate'],
            'avg_loss': self.performance_metrics['avg_loss'],
            'avg_improvement': self.performance_metrics['avg_improvement']
        }
        
    def reset_learning_state(self):
        """학습 상태 초기화"""
        self.learning_history.clear()
        self.adaptation_rate = 1.0
        self.performance_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_loss': 0.0,
            'avg_improvement': 0.0,
            'convergence_rate': 0.0
        }
        
    def set_learning_mode(self, mode: str):
        """학습 모드 설정"""
        if mode in ['online', 'batch']:
            self.learning_mode = mode
            logger.info(f"학습 모드가 {mode}로 변경되었습니다.")
        else:
            raise ValueError(f"지원하지 않는 학습 모드: {mode}")
            
    def get_optimal_learning_rate(self) -> float:
        """최적 학습률 계산"""
        if not self.learning_history:
            return self.learning_rate
            
        # 최근 성공적인 업데이트들의 학습률 평균
        recent_successful = [
            h for h in list(self.learning_history)[-50:]
            if np.linalg.norm(h['diff']) < 0.1
        ]
        
        if recent_successful:
            optimal_lr = np.mean([h['lr'] for h in recent_successful])
            return np.clip(optimal_lr, self.min_lr, self.max_lr)
        else:
            return self.learning_rate
