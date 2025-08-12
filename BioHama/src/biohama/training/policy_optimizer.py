"""
정책 최적화기 (Policy Optimizer)

A-GRPO 알고리즘의 핵심 구성 요소로, 바이오 영감 신경전달물질 시스템을 모방한
적응형 정책 최적화를 수행합니다.

주요 기능:
- 도파민 기반 보상 학습
- 세로토닌 기반 안정성 조절
- 노르에피네프린 기반 각성 조절
- 적응형 학습률 조정
- 정책 그래디언트 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from ..core.biohama_system import BioHamaSystem
from ..learning.neurotransmitter import NeurotransmitterSystem

logger = logging.getLogger(__name__)


@dataclass
class PolicyUpdate:
    """정책 업데이트 정보"""
    module_name: str
    old_policy: Dict[str, Any]
    new_policy: Dict[str, Any]
    advantage: float
    entropy: float
    kl_divergence: float
    learning_rate: float


class PolicyOptimizer:
    """정책 최적화기"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: torch.device):
        """
        정책 최적화기 초기화
        
        Args:
            config: 설정 딕셔너리
            device: 계산 디바이스
        """
        self.config = config
        self.device = device
        
        # 신경전달물질 시스템 초기화
        self.neurotransmitter_system = NeurotransmitterSystem(
            config.get('neurotransmitter', {})
        )
        
        # 정책 최적화 파라미터
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.target_kl = config.get('target_kl', 0.01)
        
        # 적응형 학습률 설정
        self.base_lr = config.get('base_learning_rate', 1e-4)
        self.lr_decay = config.get('lr_decay', 0.95)
        self.lr_min = config.get('lr_min', 1e-6)
        
        # 정책 업데이트 히스토리
        self.policy_history = []
        self.advantage_history = []
        self.entropy_history = []
        
        # 성능 추적
        self.performance_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_advantage': 0.0,
            'avg_entropy': 0.0,
            'avg_kl_divergence': 0.0
        }
        
        logger.info("정책 최적화기가 초기화되었습니다.")
        
    def optimize_policy(self,
                       biohama_system: BioHamaSystem,
                       outputs: Dict[str, Any],
                       rewards: torch.Tensor,
                       batch: Dict[str, Any]) -> float:
        """
        정책 최적화 수행
        
        Args:
            biohama_system: BioHama 시스템
            outputs: 시스템 출력
            rewards: 보상 텐서
            batch: 배치 데이터
            
        Returns:
            total_policy_loss: 총 정책 손실
        """
        total_policy_loss = 0.0
        
        # 1. 신경전달물질 수준 계산
        neurotransmitter_levels = self._calculate_neurotransmitter_levels(
            rewards=rewards,
            outputs=outputs,
            batch=batch
        )
        
        # 2. 각 모듈별 정책 최적화
        for module_name, module in biohama_system.modules.items():
            if hasattr(module, 'parameters') and module.training:
                module_loss = self._optimize_module_policy(
                    module=module,
                    module_name=module_name,
                    outputs=outputs,
                    rewards=rewards,
                    neurotransmitter_levels=neurotransmitter_levels,
                    batch=batch
                )
                
                if module_loss is not None:
                    total_policy_loss += module_loss
                    
        # 3. 신경전달물질 시스템 업데이트
        self.neurotransmitter_system.update_levels(neurotransmitter_levels)
        
        # 4. 성능 메트릭 업데이트
        self._update_performance_metrics()
        
        return total_policy_loss
        
    def _calculate_neurotransmitter_levels(self,
                                          rewards: torch.Tensor,
                                          outputs: Dict[str, Any],
                                          batch: Dict[str, Any]) -> Dict[str, float]:
        """신경전달물질 수준 계산"""
        avg_reward = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # 도파민: 보상 기반 학습
        dopamine_level = self._calculate_dopamine_level(avg_reward, reward_std)
        
        # 세로토닌: 안정성 및 만족도
        serotonin_level = self._calculate_serotonin_level(outputs, batch)
        
        # 노르에피네프린: 각성 및 주의력
        norepinephrine_level = self._calculate_norepinephrine_level(outputs, batch)
        
        return {
            'dopamine': dopamine_level,
            'serotonin': serotonin_level,
            'norepinephrine': norepinephrine_level
        }
        
    def _calculate_dopamine_level(self, 
                                 avg_reward: float, 
                                 reward_std: float) -> float:
        """도파민 수준 계산"""
        # 보상의 크기와 일관성에 기반
        reward_magnitude = np.tanh(avg_reward)  # -1 ~ 1 범위로 정규화
        reward_consistency = 1.0 / (1.0 + reward_std)  # 일관성 높을수록 높은 값
        
        dopamine = 0.7 * reward_magnitude + 0.3 * reward_consistency
        return np.clip(dopamine, 0.0, 1.0)
        
    def _calculate_serotonin_level(self,
                                  outputs: Dict[str, Any],
                                  batch: Dict[str, Any]) -> float:
        """세로토닌 수준 계산"""
        # 출력의 안정성과 일관성에 기반
        stability_score = 0.0
        
        # 어텐션 출력의 안정성
        if 'attention_output' in outputs:
            attention_output = outputs['attention_output']
            if isinstance(attention_output, torch.Tensor):
                attention_stability = 1.0 - torch.var(attention_output, dim=-1).mean().item()
                stability_score += 0.4 * attention_stability
                
        # 신뢰도의 안정성
        if 'confidence' in outputs:
            confidence = outputs['confidence']
            if isinstance(confidence, torch.Tensor):
                confidence_stability = 1.0 - torch.var(confidence).item()
                stability_score += 0.3 * confidence_stability
                
        # 품질의 일관성
        if 'quality' in outputs:
            quality = outputs['quality']
            if isinstance(quality, torch.Tensor):
                quality_consistency = 1.0 - torch.var(quality).item()
                stability_score += 0.3 * quality_consistency
                
        return np.clip(stability_score, 0.0, 1.0)
        
    def _calculate_norepinephrine_level(self,
                                       outputs: Dict[str, Any],
                                       batch: Dict[str, Any]) -> float:
        """노르에피네프린 수준 계산"""
        # 주의력과 각성 수준에 기반
        arousal_score = 0.0
        
        # 어텐션 마스크의 활성도
        if 'attention_mask' in outputs:
            attention_mask = outputs['attention_mask']
            if isinstance(attention_mask, torch.Tensor):
                attention_activity = attention_mask.float().mean().item()
                arousal_score += 0.5 * attention_activity
                
        # 종료 모듈의 활성도
        if 'should_terminate' in outputs:
            should_terminate = outputs['should_terminate']
            if isinstance(should_terminate, torch.Tensor):
                termination_activity = should_terminate.float().mean().item()
                arousal_score += 0.3 * termination_activity
                
        # 의사결정의 복잡성
        if 'decision_output' in outputs:
            decision_output = outputs['decision_output']
            if isinstance(decision_output, torch.Tensor):
                decision_entropy = F.softmax(decision_output, dim=-1)
                decision_entropy = -(decision_entropy * torch.log(decision_entropy + 1e-8)).sum(dim=-1).mean().item()
                arousal_score += 0.2 * decision_entropy
                
        return np.clip(arousal_score, 0.0, 1.0)
        
    def _optimize_module_policy(self,
                               module: nn.Module,
                               module_name: str,
                               outputs: Dict[str, Any],
                               rewards: torch.Tensor,
                               neurotransmitter_levels: Dict[str, float],
                               batch: Dict[str, Any]) -> Optional[float]:
        """모듈별 정책 최적화"""
        try:
            # 1. 현재 정책 평가
            current_policy = self._extract_module_policy(module)
            
            # 2. 어드밴티지 계산
            advantage = self._calculate_advantage(
                rewards=rewards,
                outputs=outputs,
                module_name=module_name
            )
            
            # 3. 적응형 학습률 계산
            learning_rate = self._calculate_adaptive_learning_rate(
                neurotransmitter_levels=neurotransmitter_levels,
                advantage=advantage,
                module_name=module_name
            )
            
            # 4. 정책 업데이트
            policy_loss = self._update_module_policy(
                module=module,
                module_name=module_name,
                advantage=advantage,
                learning_rate=learning_rate,
                current_policy=current_policy
            )
            
            # 5. 정책 업데이트 기록
            if policy_loss is not None:
                self._record_policy_update(
                    module_name=module_name,
                    old_policy=current_policy,
                    new_policy=self._extract_module_policy(module),
                    advantage=advantage,
                    learning_rate=learning_rate
                )
                
            return policy_loss
            
        except Exception as e:
            logger.warning(f"모듈 {module_name} 정책 최적화 실패: {e}")
            return None
            
    def _extract_module_policy(self, module: nn.Module) -> Dict[str, Any]:
        """모듈의 현재 정책 추출"""
        policy = {}
        
        # 모듈 타입별 정책 추출
        if hasattr(module, 'module_type'):
            module_type = module.module_type
            
            if module_type == 'sparse_attention':
                # 희소 어텐션 정책
                if hasattr(module, 'sparsity_ratio'):
                    policy['sparsity_ratio'] = module.sparsity_ratio
                if hasattr(module, 'pattern_learner'):
                    policy['pattern_weights'] = module.pattern_learner.pattern_weights.data.clone()
                    
            elif module_type == 'termination':
                # 종료 모듈 정책
                if hasattr(module, 'confidence_threshold'):
                    policy['confidence_threshold'] = module.confidence_threshold
                if hasattr(module, 'quality_threshold'):
                    policy['quality_threshold'] = module.quality_threshold
                    
            elif module_type == 'decision_engine':
                # 의사결정 엔진 정책
                if hasattr(module, 'policy_network'):
                    policy['policy_params'] = {
                        name: param.data.clone()
                        for name, param in module.policy_network.named_parameters()
                    }
                    
        # 일반적인 모듈 파라미터
        policy['module_params'] = {
            name: param.data.clone()
            for name, param in module.named_parameters()
        }
        
        return policy
        
    def _calculate_advantage(self,
                           rewards: torch.Tensor,
                           outputs: Dict[str, Any],
                           module_name: str) -> float:
        """어드밴티지 계산"""
        # 기본 어드밴티지: 보상의 평균
        base_advantage = rewards.mean().item()
        
        # 모듈별 특화 어드밴티지
        module_advantage = 0.0
        
        if module_name == 'sparse_attention':
            # 어텐션 효율성 기반 어드밴티지
            if 'computation_savings' in outputs:
                module_advantage = outputs['computation_savings'] * 0.3
                
        elif module_name == 'termination':
            # 종료 정확성 기반 어드밴티지
            if 'confidence' in outputs and 'quality' in outputs:
                confidence = outputs['confidence']
                quality = outputs['quality']
                if isinstance(confidence, torch.Tensor) and isinstance(quality, torch.Tensor):
                    correlation = torch.corrcoef(
                        torch.stack([confidence.flatten(), quality.flatten()])
                    )[0, 1].item()
                    module_advantage = correlation * 0.2
                    
        elif module_name == 'decision_engine':
            # 의사결정 품질 기반 어드밴티지
            if 'decision_output' in outputs:
                decision_output = outputs['decision_output']
                if isinstance(decision_output, torch.Tensor):
                    decision_confidence = F.softmax(decision_output, dim=-1).max(dim=-1)[0].mean().item()
                    module_advantage = decision_confidence * 0.2
                    
        return base_advantage + module_advantage
        
    def _calculate_adaptive_learning_rate(self,
                                        neurotransmitter_levels: Dict[str, float],
                                        advantage: float,
                                        module_name: str) -> float:
        """적응형 학습률 계산"""
        # 기본 학습률
        lr = self.base_lr
        
        # 도파민 기반 학습률 조정
        dopamine_level = neurotransmitter_levels['dopamine']
        lr *= (1.0 + dopamine_level * 0.5)  # 도파민 높을수록 학습률 증가
        
        # 세로토닌 기반 안정성 조정
        serotonin_level = neurotransmitter_levels['serotonin']
        lr *= (1.0 - serotonin_level * 0.3)  # 세로토닌 높을수록 안정적 학습
        
        # 노르에피네프린 기반 각성 조정
        norepinephrine_level = neurotransmitter_levels['norepinephrine']
        lr *= (1.0 + norepinephrine_level * 0.2)  # 각성 높을수록 적극적 학습
        
        # 어드밴티지 기반 조정
        if advantage > 0:
            lr *= (1.0 + advantage * 0.1)  # 긍정적 어드밴티지일 때 학습률 증가
        else:
            lr *= (1.0 - abs(advantage) * 0.05)  # 부정적 어드밴티지일 때 학습률 감소
            
        # 모듈별 특화 조정
        if module_name == 'sparse_attention':
            lr *= 1.2  # 어텐션 모듈은 더 적극적 학습
        elif module_name == 'termination':
            lr *= 0.8  # 종료 모듈은 보수적 학습
            
        # 학습률 범위 제한
        lr = np.clip(lr, self.lr_min, self.base_lr * 2.0)
        
        return lr
        
    def _update_module_policy(self,
                             module: nn.Module,
                             module_name: str,
                             advantage: float,
                             learning_rate: float,
                             current_policy: Dict[str, Any]) -> Optional[float]:
        """모듈 정책 업데이트"""
        try:
            # 1. 정책 그래디언트 계산
            policy_loss = self._calculate_policy_loss(
                module=module,
                advantage=advantage,
                module_name=module_name
            )
            
            if policy_loss is None:
                return None
                
            # 2. 그래디언트 계산
            module.zero_grad()
            policy_loss.backward()
            
            # 3. 그래디언트 클리핑
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    self.max_grad_norm
                )
                
            # 4. 파라미터 업데이트 (수동으로 학습률 적용)
            with torch.no_grad():
                for param in module.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad
                        
            # 5. KL 발산 계산
            new_policy = self._extract_module_policy(module)
            kl_divergence = self._calculate_kl_divergence(
                old_policy=current_policy,
                new_policy=new_policy
            )
            
            # 6. KL 발산 기반 조기 종료
            if kl_divergence > self.target_kl:
                logger.info(f"모듈 {module_name}: KL 발산 임계값 초과 ({kl_divergence:.4f})")
                # 파라미터 롤백
                self._rollback_policy(module, current_policy)
                return None
                
            return policy_loss.item()
            
        except Exception as e:
            logger.error(f"모듈 {module_name} 정책 업데이트 실패: {e}")
            return None
            
    def _calculate_policy_loss(self,
                              module: nn.Module,
                              advantage: float,
                              module_name: str) -> Optional[torch.Tensor]:
        """정책 손실 계산"""
        try:
            # 모듈 타입별 손실 계산
            if hasattr(module, 'module_type'):
                module_type = module.module_type
                
                if module_type == 'sparse_attention':
                    return self._calculate_sparse_attention_loss(module, advantage)
                elif module_type == 'termination':
                    return self._calculate_termination_loss(module, advantage)
                elif module_type == 'decision_engine':
                    return self._calculate_decision_engine_loss(module, advantage)
                    
            # 기본 정책 손실
            return self._calculate_default_policy_loss(module, advantage)
            
        except Exception as e:
            logger.warning(f"정책 손실 계산 실패: {e}")
            return None
            
    def _calculate_sparse_attention_loss(self,
                                        module: nn.Module,
                                        advantage: float) -> torch.Tensor:
        """희소 어텐션 정책 손실"""
        # 어텐션 품질과 효율성의 균형
        attention_quality = torch.var(module.pattern_learner.pattern_weights).mean()
        sparsity_efficiency = torch.mean(module.mask_generator.sparsity_ratio)
        
        # 어드밴티지 기반 손실
        advantage_loss = -advantage * torch.log(torch.sigmoid(attention_quality + sparsity_efficiency))
        
        # 정규화 항
        regularization = 0.01 * (attention_quality + sparsity_efficiency)
        
        return advantage_loss + regularization
        
    def _calculate_termination_loss(self,
                                   module: nn.Module,
                                   advantage: float) -> torch.Tensor:
        """종료 모듈 정책 손실"""
        # 신뢰도와 품질의 균형
        confidence_quality = torch.var(module.confidence_estimator.confidence_heads[0].weight).mean()
        
        # 어드밴티지 기반 손실
        advantage_loss = -advantage * torch.log(torch.sigmoid(confidence_quality))
        
        # 정규화 항
        regularization = 0.01 * confidence_quality
        
        return advantage_loss + regularization
        
    def _calculate_decision_engine_loss(self,
                                       module: nn.Module,
                                       advantage: float) -> torch.Tensor:
        """의사결정 엔진 정책 손실"""
        # 의사결정 품질
        if hasattr(module, 'policy_network'):
            decision_quality = torch.var(module.policy_network.weight).mean()
        else:
            decision_quality = torch.tensor(0.1)
            
        # 어드밴티지 기반 손실
        advantage_loss = -advantage * torch.log(torch.sigmoid(decision_quality))
        
        # 정규화 항
        regularization = 0.01 * decision_quality
        
        return advantage_loss + regularization
        
    def _calculate_default_policy_loss(self,
                                      module: nn.Module,
                                      advantage: float) -> torch.Tensor:
        """기본 정책 손실"""
        # 모듈 파라미터의 엔트로피
        entropy = 0.0
        for param in module.parameters():
            if param.requires_grad:
                entropy += torch.var(param).mean()
                
        # 어드밴티지 기반 손실
        advantage_loss = -advantage * torch.log(torch.sigmoid(entropy))
        
        # 정규화 항
        regularization = 0.01 * entropy
        
        return advantage_loss + regularization
        
    def _calculate_kl_divergence(self,
                                old_policy: Dict[str, Any],
                                new_policy: Dict[str, Any]) -> float:
        """KL 발산 계산"""
        total_kl = 0.0
        count = 0
        
        # 모듈 파라미터 비교
        if 'module_params' in old_policy and 'module_params' in new_policy:
            for name in old_policy['module_params']:
                if name in new_policy['module_params']:
                    old_param = old_policy['module_params'][name]
                    new_param = new_policy['module_params'][name]
                    
                    # KL 발산 계산
                    kl = F.kl_div(
                        F.log_softmax(new_param.flatten(), dim=0),
                        F.softmax(old_param.flatten(), dim=0),
                        reduction='batchmean'
                    )
                    
                    total_kl += kl.item()
                    count += 1
                    
        return total_kl / max(count, 1)
        
    def _rollback_policy(self,
                        module: nn.Module,
                        old_policy: Dict[str, Any]):
        """정책 롤백"""
        if 'module_params' in old_policy:
            for name, param in module.named_parameters():
                if name in old_policy['module_params']:
                    param.data = old_policy['module_params'][name].clone()
                    
    def _record_policy_update(self,
                             module_name: str,
                             old_policy: Dict[str, Any],
                             new_policy: Dict[str, Any],
                             advantage: float,
                             learning_rate: float):
        """정책 업데이트 기록"""
        # 엔트로피 계산
        entropy = 0.0
        if 'module_params' in new_policy:
            for param in new_policy['module_params'].values():
                entropy += torch.var(param).item()
                
        # KL 발산 계산
        kl_divergence = self._calculate_kl_divergence(old_policy, new_policy)
        
        # 업데이트 기록
        policy_update = PolicyUpdate(
            module_name=module_name,
            old_policy=old_policy,
            new_policy=new_policy,
            advantage=advantage,
            entropy=entropy,
            kl_divergence=kl_divergence,
            learning_rate=learning_rate
        )
        
        self.policy_history.append(policy_update)
        
        # 히스토리 크기 제한
        if len(self.policy_history) > 1000:
            self.policy_history = self.policy_history[-500:]
            
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        if not self.policy_history:
            return
            
        recent_updates = self.policy_history[-100:]
        
        self.performance_metrics['total_updates'] += len(recent_updates)
        self.performance_metrics['successful_updates'] += len([
            u for u in recent_updates if u.advantage > 0
        ])
        
        self.performance_metrics['avg_advantage'] = np.mean([
            u.advantage for u in recent_updates
        ])
        
        self.performance_metrics['avg_entropy'] = np.mean([
            u.entropy for u in recent_updates
        ])
        
        self.performance_metrics['avg_kl_divergence'] = np.mean([
            u.kl_divergence for u in recent_updates
        ])
        
    def get_policy_summary(self) -> Dict[str, Any]:
        """정책 최적화 요약 정보"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'neurotransmitter_levels': self.neurotransmitter_system.get_current_levels(),
            'recent_advantages': [
                u.advantage for u in self.policy_history[-10:]
            ],
            'recent_learning_rates': [
                u.learning_rate for u in self.policy_history[-10:]
            ]
        }
        
    def reset_metrics(self):
        """메트릭 초기화"""
        self.performance_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_advantage': 0.0,
            'avg_entropy': 0.0,
            'avg_kl_divergence': 0.0
        }
        
        self.policy_history = []
        self.advantage_history = []
        self.entropy_history = []
