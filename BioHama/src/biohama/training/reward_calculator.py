"""
보상 계산기 (Reward Calculator)

다양한 보상 신호를 통합하여 최적의 보상을 계산하는 시스템입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AccuracyReward:
    """정확도 보상 계산기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accuracy_threshold = config.get('accuracy_threshold', 0.8)
        self.reward_scale = config.get('reward_scale', 1.0)
        
    def calculate(self, 
                 outputs: Dict[str, Any],
                 targets: Optional[torch.Tensor]) -> float:
        """정확도 기반 보상 계산"""
        if targets is None:
            return 0.0
            
        accuracy = 0.0
        
        # 의사결정 출력의 정확도
        if 'decision_output' in outputs:
            decision_output = outputs['decision_output']
            if isinstance(decision_output, torch.Tensor):
                predictions = torch.argmax(decision_output, dim=-1)
                accuracy = (predictions == targets).float().mean().item()
                
        # 어텐션 출력의 정확도 (간접적)
        elif 'attention_output' in outputs:
            attention_output = outputs['attention_output']
            if isinstance(attention_output, torch.Tensor):
                # 어텐션 출력을 분류로 변환
                attention_logits = attention_output.mean(dim=1)
                predictions = torch.argmax(attention_logits, dim=-1)
                accuracy = (predictions == targets).float().mean().item()
                
        # 보상 계산
        reward = accuracy * self.reward_scale
        
        # 임계값 기반 보너스
        if accuracy >= self.accuracy_threshold:
            reward += 0.2 * self.reward_scale
            
        return reward


class EfficiencyReward:
    """효율성 보상 계산기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.efficiency_threshold = config.get('efficiency_threshold', 0.7)
        self.reward_scale = config.get('reward_scale', 0.5)
        
    def calculate(self, outputs: Dict[str, Any]) -> float:
        """효율성 기반 보상 계산"""
        efficiency_score = 0.0
        
        # 계산 효율성 (희소 어텐션)
        if 'computation_savings' in outputs:
            savings = outputs['computation_savings']
            if isinstance(savings, (float, int)):
                efficiency_score += savings
                
        # 어텐션 효율성
        if 'attention_mask' in outputs:
            attention_mask = outputs['attention_mask']
            if isinstance(attention_mask, torch.Tensor):
                # 어텐션 마스크의 희소성
                sparsity = 1.0 - attention_mask.float().mean().item()
                efficiency_score += sparsity * 0.3
                
        # 종료 효율성
        if 'should_terminate' in outputs:
            should_terminate = outputs['should_terminate']
            if isinstance(should_terminate, torch.Tensor):
                # 적절한 종료 비율
                termination_rate = should_terminate.float().mean().item()
                efficiency_score += termination_rate * 0.2
                
        # 보상 계산
        reward = efficiency_score * self.reward_scale
        
        # 임계값 기반 보너스
        if efficiency_score >= self.efficiency_threshold:
            reward += 0.1 * self.reward_scale
            
        return reward


class ConsistencyReward:
    """일관성 보상 계산기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consistency_threshold = config.get('consistency_threshold', 0.6)
        self.reward_scale = config.get('reward_scale', 0.3)
        
    def calculate(self, outputs: Dict[str, Any]) -> float:
        """일관성 기반 보상 계산"""
        consistency_score = 0.0
        
        # 어텐션 일관성
        if 'attention_output' in outputs:
            attention_output = outputs['attention_output']
            if isinstance(attention_output, torch.Tensor):
                # 어텐션 출력의 분산 (낮을수록 일관성 높음)
                attention_variance = torch.var(attention_output, dim=-1).mean().item()
                consistency_score += (1.0 - attention_variance) * 0.4
                
        # 신뢰도 일관성
        if 'confidence' in outputs:
            confidence = outputs['confidence']
            if isinstance(confidence, torch.Tensor):
                # 신뢰도의 분산 (낮을수록 일관성 높음)
                confidence_variance = torch.var(confidence).item()
                consistency_score += (1.0 - confidence_variance) * 0.3
                
        # 품질 일관성
        if 'quality' in outputs:
            quality = outputs['quality']
            if isinstance(quality, torch.Tensor):
                # 품질의 분산 (낮을수록 일관성 높음)
                quality_variance = torch.var(quality).item()
                consistency_score += (1.0 - quality_variance) * 0.3
                
        # 보상 계산
        reward = consistency_score * self.reward_scale
        
        # 임계값 기반 보너스
        if consistency_score >= self.consistency_threshold:
            reward += 0.05 * self.reward_scale
            
        return reward


class RewardCalculator:
    """보상 계산기 메인 클래스"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # 보상 구성 요소들
        self.accuracy_reward = AccuracyReward(
            config.get('accuracy_reward', {})
        )
        
        self.efficiency_reward = EfficiencyReward(
            config.get('efficiency_reward', {})
        )
        
        self.consistency_reward = ConsistencyReward(
            config.get('consistency_reward', {})
        )
        
        # 보상 가중치
        self.reward_weights = config.get('reward_weights', {
            'accuracy': 0.5,
            'efficiency': 0.3,
            'consistency': 0.2
        })
        
        # 보상 통계
        self.reward_stats = {
            'total_calculations': 0,
            'avg_accuracy_reward': 0.0,
            'avg_efficiency_reward': 0.0,
            'avg_consistency_reward': 0.0,
            'avg_total_reward': 0.0
        }
        
        logger.info("보상 계산기가 초기화되었습니다.")
        
    def calculate_rewards(self,
                         outputs: Dict[str, Any],
                         targets: Optional[torch.Tensor] = None,
                         preferences: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        통합 보상 계산
        
        Args:
            outputs: 시스템 출력
            targets: 목표값
            preferences: 사용자 선호도
            
        Returns:
            rewards: 계산된 보상 텐서
        """
        # 개별 보상 계산
        accuracy_reward = self.accuracy_reward.calculate(outputs, targets)
        efficiency_reward = self.efficiency_reward.calculate(outputs)
        consistency_reward = self.consistency_reward.calculate(outputs)
        
        # 선호도 기반 가중치 조정
        adjusted_weights = self._adjust_weights_by_preferences(preferences)
        
        # 가중 보상 계산
        total_reward = (
            accuracy_reward * adjusted_weights['accuracy'] +
            efficiency_reward * adjusted_weights['efficiency'] +
            consistency_reward * adjusted_weights['consistency']
        )
        
        # 보상 정규화
        normalized_reward = self._normalize_reward(total_reward)
        
        # 배치 크기에 맞게 확장
        batch_size = self._get_batch_size(outputs)
        rewards = torch.full((batch_size,), normalized_reward, device=self.device)
        
        # 통계 업데이트
        self._update_stats(accuracy_reward, efficiency_reward, consistency_reward, normalized_reward)
        
        return rewards
        
    def _adjust_weights_by_preferences(self, 
                                     preferences: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """선호도 기반 가중치 조정"""
        weights = self.reward_weights.copy()
        
        if preferences is not None:
            # 선호도에 따른 가중치 조정
            for key, value in preferences.items():
                if key in weights:
                    # 선호도 값을 0.5~1.5 범위로 조정
                    adjustment = 0.5 + value
                    weights[key] *= adjustment
                elif key in ['accuracy', 'efficiency', 'consistency']:
                    # 보상 구성 요소에 대한 직접적인 선호도
                    adjustment = 0.5 + value
                    weights[key] *= adjustment
                    
            # 가중치 정규화
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
        return weights
        
    def update_preferences(self, preference_model):
        """선호도 모델로부터 선호도 정보 업데이트"""
        if hasattr(preference_model, 'memory'):
            # 메모리에서 최근 선호도 데이터 추출
            recent_explicit = preference_model.memory.retrieve_by_type('explicit', limit=10)
            recent_implicit = preference_model.memory.retrieve_by_type('implicit', limit=10)
            
            # 선호도 통계 계산
            explicit_prefs = {}
            implicit_prefs = {}
            
            for _, data in recent_explicit:
                if isinstance(data.preference_value, (int, float)):
                    key = data.context.get('key', 'unknown') if data.context else 'unknown'
                    if key not in explicit_prefs:
                        explicit_prefs[key] = []
                    explicit_prefs[key].append(data.preference_value)
                    
            for _, data in recent_implicit:
                if isinstance(data.preference_value, (int, float)):
                    key = data.context.get('key', 'unknown') if data.context else 'unknown'
                    if key not in implicit_prefs:
                        implicit_prefs[key] = []
                    implicit_prefs[key].append(data.preference_value)
                    
            # 평균 선호도 계산
            avg_explicit = {}
            for key, values in explicit_prefs.items():
                avg_explicit[key] = np.mean(values)
                
            avg_implicit = {}
            for key, values in implicit_prefs.items():
                avg_implicit[key] = np.mean(values)
                
            # 보상 가중치 업데이트
            self._update_weights_from_preferences(avg_explicit, avg_implicit)
            
    def _update_weights_from_preferences(self, explicit_prefs: Dict[str, float], implicit_prefs: Dict[str, float]):
        """선호도 정보로부터 보상 가중치 업데이트"""
        # 명시적 선호도 기반 조정
        for key, value in explicit_prefs.items():
            if key in self.reward_weights:
                # 명시적 선호도는 더 큰 가중치
                adjustment = 0.3 + value * 0.7
                self.reward_weights[key] *= adjustment
                
        # 암시적 선호도 기반 조정
        for key, value in implicit_prefs.items():
            if key in self.reward_weights:
                # 암시적 선호도는 작은 가중치
                adjustment = 0.7 + value * 0.3
                self.reward_weights[key] *= adjustment
                
        # 가중치 정규화
        total_weight = sum(self.reward_weights.values())
        self.reward_weights = {k: v / total_weight for k, v in self.reward_weights.items()}
        
    def _normalize_reward(self, reward: float) -> float:
        """보상 정규화"""
        # 0~1 범위로 정규화
        normalized = np.clip(reward, 0.0, 1.0)
        
        # 로그 스케일링 (선택사항)
        if self.config.get('use_log_scaling', False):
            normalized = np.log(1 + normalized)
            
        return normalized
        
    def _get_batch_size(self, outputs: Dict[str, Any]) -> int:
        """배치 크기 추출"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 1
        
    def _update_stats(self,
                     accuracy_reward: float,
                     efficiency_reward: float,
                     consistency_reward: float,
                     total_reward: float):
        """통계 업데이트"""
        self.reward_stats['total_calculations'] += 1
        
        # 이동 평균 업데이트
        alpha = 0.1  # 학습률
        n = self.reward_stats['total_calculations']
        
        self.reward_stats['avg_accuracy_reward'] = (
            (self.reward_stats['avg_accuracy_reward'] * (n - 1) + accuracy_reward) / n
        )
        
        self.reward_stats['avg_efficiency_reward'] = (
            (self.reward_stats['avg_efficiency_reward'] * (n - 1) + efficiency_reward) / n
        )
        
        self.reward_stats['avg_consistency_reward'] = (
            (self.reward_stats['avg_consistency_reward'] * (n - 1) + consistency_reward) / n
        )
        
        self.reward_stats['avg_total_reward'] = (
            (self.reward_stats['avg_total_reward'] * (n - 1) + total_reward) / n
        )
        
    def get_reward_summary(self) -> Dict[str, Any]:
        """보상 계산 요약 정보"""
        return {
            'reward_stats': self.reward_stats.copy(),
            'reward_weights': self.reward_weights.copy(),
            'config': {
                'accuracy_threshold': self.accuracy_reward.accuracy_threshold,
                'efficiency_threshold': self.efficiency_reward.efficiency_threshold,
                'consistency_threshold': self.consistency_reward.consistency_threshold
            }
        }
        
    def reset_stats(self):
        """통계 초기화"""
        self.reward_stats = {
            'total_calculations': 0,
            'avg_accuracy_reward': 0.0,
            'avg_efficiency_reward': 0.0,
            'avg_consistency_reward': 0.0,
            'avg_total_reward': 0.0
        }
