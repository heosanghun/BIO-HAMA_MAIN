"""
A-GRPO 훈련 시스템

바이오 영감 적응형 정책 최적화 알고리즘을 사용하여 BioHama 시스템을 훈련시키는
메인 훈련 시스템입니다.

주요 기능:
- A-GRPO 알고리즘 구현
- 다중 모듈 동시 훈련
- 적응형 학습률 조정
- 성능 모니터링 및 로깅
- 체크포인트 관리
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import json
import os
from datetime import datetime
import time
from pathlib import Path

from ..core.biohama_system import BioHamaSystem
from .policy_optimizer import PolicyOptimizer
from .preference_model import PreferenceModel
from .reward_calculator import RewardCalculator
from ..utils.logging import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class AGRPOTrainer:
    """A-GRPO 훈련기"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 biohama_system: Optional[BioHamaSystem] = None):
        """
        A-GRPO 훈련기 초기화
        
        Args:
            config: 훈련 설정
            biohama_system: 훈련할 BioHama 시스템 (None이면 새로 생성)
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # BioHama 시스템 초기화
        if biohama_system is None:
            self.biohama_system = BioHamaSystem(config.get('system_config', {}))
        else:
            self.biohama_system = biohama_system
            
        self.biohama_system.to(self.device)
        
        # 훈련 구성 요소 초기화
        self._initialize_training_components()
        
        # 훈련 상태 초기화
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_reward': -float('inf'),
            'best_accuracy': 0.0,
            'training_losses': [],
            'validation_losses': [],
            'rewards': [],
            'learning_rates': [],
            'convergence_history': []
        }
        
        # 로깅 설정
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # 체크포인트 설정
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"A-GRPO 훈련기가 초기화되었습니다. 디바이스: {self.device}")
        
    def _initialize_training_components(self):
        """훈련 구성 요소 초기화"""
        # 정책 최적화기
        self.policy_optimizer = PolicyOptimizer(
            config=self.config.get('policy_optimizer', {}),
            device=self.device
        )
        
        # 선호도 모델
        self.preference_model = PreferenceModel(
            config=self.config.get('preference_model', {}),
            device=self.device
        )
        
        # 보상 계산기
        self.reward_calculator = RewardCalculator(
            config=self.config.get('reward_calculator', {}),
            device=self.device
        )
        
        # 옵티마이저들
        self.optimizers = {}
        self.schedulers = {}
        
        # 각 모듈별 옵티마이저 설정
        for module_name, module in self.biohama_system.modules.items():
            if hasattr(module, 'parameters'):
                optimizer_config = self.config.get('optimizers', {}).get(module_name, {})
                
                optimizer_type = optimizer_config.get('type', 'adam')
                lr = optimizer_config.get('learning_rate', 1e-4)
                weight_decay = optimizer_config.get('weight_decay', 1e-5)
                
                if optimizer_type == 'adam':
                    optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
                elif optimizer_type == 'adamw':
                    optimizer = optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
                elif optimizer_type == 'sgd':
                    optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
                else:
                    raise ValueError(f"지원하지 않는 옵티마이저 타입: {optimizer_type}")
                
                self.optimizers[module_name] = optimizer
                
                # 스케줄러 설정
                scheduler_config = optimizer_config.get('scheduler', {})
                scheduler_type = scheduler_config.get('type', 'cosine')
                
                if scheduler_type == 'cosine':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=scheduler_config.get('T_max', 1000),
                        eta_min=scheduler_config.get('eta_min', 1e-6)
                    )
                elif scheduler_type == 'step':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=scheduler_config.get('step_size', 100),
                        gamma=scheduler_config.get('gamma', 0.9)
                    )
                elif scheduler_type == 'plateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='max',
                        factor=scheduler_config.get('factor', 0.5),
                        patience=scheduler_config.get('patience', 10),
                        verbose=True
                    )
                else:
                    scheduler = None
                    
                self.schedulers[module_name] = scheduler
                
        logger.info(f"{len(self.optimizers)}개의 모듈 옵티마이저가 초기화되었습니다.")
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """
        한 에포크 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            epoch: 현재 에포크 번호
            
        Returns:
            epoch_metrics: 에포크 메트릭
        """
        self.biohama_system.train()
        
        epoch_losses = []
        epoch_rewards = []
        epoch_accuracies = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 배치를 디바이스로 이동
            batch = self._move_batch_to_device(batch)
            
            # 훈련 스텝 실행
            step_metrics = self._train_step(batch, epoch, batch_idx)
            
            epoch_losses.append(step_metrics['loss'])
            epoch_rewards.append(step_metrics['reward'])
            epoch_accuracies.append(step_metrics['accuracy'])
            
            # 로깅
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self._log_training_step(epoch, batch_idx, step_metrics)
                
        # 에포크 메트릭 계산
        epoch_metrics = {
            'epoch': epoch,
            'avg_loss': np.mean(epoch_losses),
            'avg_reward': np.mean(epoch_rewards),
            'avg_accuracy': np.mean(epoch_accuracies),
            'std_loss': np.std(epoch_losses),
            'std_reward': np.std(epoch_rewards)
        }
        
        # 훈련 상태 업데이트
        self.training_state['training_losses'].append(epoch_metrics['avg_loss'])
        self.training_state['rewards'].append(epoch_metrics['avg_reward'])
        
        return epoch_metrics
        
    def _train_step(self, 
                   batch: Dict[str, Any],
                   epoch: int,
                   step: int) -> Dict[str, float]:
        """
        단일 훈련 스텝
        
        Args:
            batch: 배치 데이터
            epoch: 에포크 번호
            step: 스텝 번호
            
        Returns:
            step_metrics: 스텝 메트릭
        """
        # 1. 순전파
        outputs = self.biohama_system.process_input(batch['input'])
        
        # 2. 보상 계산
        rewards = self.reward_calculator.calculate_rewards(
            outputs=outputs,
            targets=batch.get('target'),
            preferences=batch.get('preferences')
        )
        
        # 3. 선호도 모델 업데이트
        preference_loss = self.preference_model.update(
            outputs=outputs,
            preferences=batch.get('preferences'),
            rewards=rewards
        )
        
        # 4. 정책 최적화
        policy_loss = self.policy_optimizer.optimize_policy(
            biohama_system=self.biohama_system,
            outputs=outputs,
            rewards=rewards,
            batch=batch
        )
        
        # 5. 모듈별 손실 계산 및 역전파
        total_loss = 0.0
        module_losses = {}
        
        for module_name, module in self.biohama_system.modules.items():
            if module_name in self.optimizers:
                optimizer = self.optimizers[module_name]
                
                # 모듈별 손실 계산
                module_loss = self._calculate_module_loss(
                    module=module,
                    outputs=outputs,
                    targets=batch.get('target'),
                    rewards=rewards
                )
                
                if module_loss is not None:
                    optimizer.zero_grad()
                    module_loss.backward()
                    
                    # 그래디언트 클리핑
                    if self.config.get('gradient_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    optimizer.step()
                    
                    total_loss += module_loss.item()
                    module_losses[module_name] = module_loss.item()
        
        # 6. 스케줄러 업데이트
        for module_name, scheduler in self.schedulers.items():
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(rewards.mean().item())
                else:
                    scheduler.step()
        
        # 7. 메트릭 계산
        accuracy = self._calculate_accuracy(outputs, batch.get('target'))
        avg_reward = rewards.mean().item()
        
        step_metrics = {
            'loss': total_loss,
            'reward': avg_reward,
            'accuracy': accuracy,
            'preference_loss': preference_loss,
            'policy_loss': policy_loss,
            'module_losses': module_losses
        }
        
        # 훈련 상태 업데이트
        self.training_state['step'] += 1
        
        return step_metrics
        
    def _calculate_module_loss(self,
                              module: nn.Module,
                              outputs: Dict[str, Any],
                              targets: Optional[torch.Tensor],
                              rewards: torch.Tensor) -> Optional[torch.Tensor]:
        """모듈별 손실 계산"""
        module_name = getattr(module, 'name', 'unknown')
        
        # 모듈 타입에 따른 손실 계산
        if hasattr(module, 'module_type'):
            module_type = module.module_type
            
            if module_type == 'sparse_attention':
                # 희소 어텐션 손실: 어텐션 품질과 계산 효율성
                attention_output = outputs.get('attention_output')
                if attention_output is not None:
                    # 어텐션 출력의 일관성 손실
                    consistency_loss = torch.var(attention_output, dim=-1).mean()
                    # 희소성 보장 손실
                    sparsity_loss = torch.mean(outputs.get('attention_mask', torch.ones_like(attention_output)))
                    return consistency_loss + 0.1 * sparsity_loss
                    
            elif module_type == 'termination':
                # 종료 모듈 손실: 신뢰도와 품질의 균형
                confidence = outputs.get('confidence')
                quality = outputs.get('quality')
                if confidence is not None and quality is not None:
                    # 신뢰도와 품질의 상관관계 손실
                    correlation_loss = 1 - torch.corrcoef(
                        torch.stack([confidence.flatten(), quality.flatten()])
                    )[0, 1]
                    return torch.abs(correlation_loss)
                    
            elif module_type == 'decision_engine':
                # 의사결정 엔진 손실: 의사결정 품질
                if targets is not None:
                    decision_output = outputs.get('decision_output')
                    if decision_output is not None:
                        return nn.functional.cross_entropy(decision_output, targets)
                        
        # 기본 손실: 보상 기반
        return -rewards.mean()
        
    def _calculate_accuracy(self,
                           outputs: Dict[str, Any],
                           targets: Optional[torch.Tensor]) -> float:
        """정확도 계산"""
        if targets is None:
            return 0.0
            
        # 출력에서 예측 추출
        predictions = None
        
        if 'decision_output' in outputs:
            predictions = torch.argmax(outputs['decision_output'], dim=-1)
        elif 'attention_output' in outputs:
            # 어텐션 출력을 분류로 변환
            predictions = torch.argmax(outputs['attention_output'].mean(dim=1), dim=-1)
        else:
            return 0.0
            
        if predictions is not None and targets is not None:
            return (predictions == targets).float().mean().item()
            
        return 0.0
        
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """배치를 디바이스로 이동"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = self._move_batch_to_device(value)
            elif isinstance(value, list):
                device_batch[key] = [
                    item.to(self.device) if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            else:
                device_batch[key] = value
                
        return device_batch
        
    def _log_training_step(self,
                          epoch: int,
                          step: int,
                          metrics: Dict[str, float]):
        """훈련 스텝 로깅"""
        log_msg = f"Epoch {epoch}, Step {step}: "
        log_msg += f"Loss={metrics['loss']:.4f}, "
        log_msg += f"Reward={metrics['reward']:.4f}, "
        log_msg += f"Accuracy={metrics['accuracy']:.4f}"
        
        logger.info(log_msg)
        
    def validate(self, 
                val_loader: DataLoader) -> Dict[str, float]:
        """
        검증 실행
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            val_metrics: 검증 메트릭
        """
        self.biohama_system.eval()
        
        val_losses = []
        val_rewards = []
        val_accuracies = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                # 순전파
                outputs = self.biohama_system.process_input(batch['input'])
                
                # 보상 계산
                rewards = self.reward_calculator.calculate_rewards(
                    outputs=outputs,
                    targets=batch.get('target'),
                    preferences=batch.get('preferences')
                )
                
                # 손실 계산
                loss = self._calculate_validation_loss(outputs, batch, rewards)
                
                # 정확도 계산
                accuracy = self._calculate_accuracy(outputs, batch.get('target'))
                
                val_losses.append(loss)
                val_rewards.append(rewards.mean().item())
                val_accuracies.append(accuracy)
                
        val_metrics = {
            'val_loss': np.mean(val_losses),
            'val_reward': np.mean(val_rewards),
            'val_accuracy': np.mean(val_accuracies),
            'val_std_loss': np.std(val_losses),
            'val_std_reward': np.std(val_rewards)
        }
        
        # 검증 상태 업데이트
        self.training_state['validation_losses'].append(val_metrics['val_loss'])
        
        return val_metrics
        
    def _calculate_validation_loss(self,
                                  outputs: Dict[str, Any],
                                  batch: Dict[str, Any],
                                  rewards: torch.Tensor) -> float:
        """검증 손실 계산"""
        total_loss = 0.0
        
        for module in self.biohama_system.modules.values():
            module_loss = self._calculate_module_loss(
                module=module,
                outputs=outputs,
                targets=batch.get('target'),
                rewards=rewards
            )
            
            if module_loss is not None:
                total_loss += module_loss.item()
                
        return total_loss
        
    def save_checkpoint(self, 
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'training_state': self.training_state,
            'biohama_system_state': self.biohama_system.state_dict(),
            'optimizers_state': {
                name: optimizer.state_dict() 
                for name, optimizer in self.optimizers.items()
            },
            'schedulers_state': {
                name: scheduler.state_dict() if scheduler is not None else None
                for name, scheduler in self.schedulers.items()
            },
            'metrics': metrics,
            'config': self.config
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"최고 성능 체크포인트 저장: {best_path}")
            
        logger.info(f"체크포인트 저장: {checkpoint_path}")
        
    def load_checkpoint(self, 
                       checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # BioHama 시스템 상태 복원
        self.biohama_system.load_state_dict(checkpoint['biohama_system_state'])
        
        # 옵티마이저 상태 복원
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizers_state']:
                optimizer.load_state_dict(checkpoint['optimizers_state'][name])
                
        # 스케줄러 상태 복원
        for name, scheduler in self.schedulers.items():
            if name in checkpoint['schedulers_state']:
                scheduler_state = checkpoint['schedulers_state'][name]
                if scheduler is not None and scheduler_state is not None:
                    scheduler.load_state_dict(scheduler_state)
                    
        # 훈련 상태 복원
        self.training_state = checkpoint['training_state']
        
        logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
        return checkpoint['metrics']
        
    def get_training_summary(self) -> Dict[str, Any]:
        """훈련 요약 정보 반환"""
        return {
            'current_epoch': self.training_state['epoch'],
            'total_steps': self.training_state['step'],
            'best_reward': self.training_state['best_reward'],
            'best_accuracy': self.training_state['best_accuracy'],
            'current_lr': {
                name: optimizer.param_groups[0]['lr']
                for name, optimizer in self.optimizers.items()
            },
            'convergence_status': self._analyze_convergence()
        }
        
    def _analyze_convergence(self) -> Dict[str, Any]:
        """수렴 상태 분석"""
        if len(self.training_state['training_losses']) < 10:
            return {'status': 'insufficient_data'}
            
        recent_losses = self.training_state['training_losses'][-10:]
        recent_rewards = self.training_state['rewards'][-10:]
        
        # 손실 감소율
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # 보상 증가율
        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        # 변동성
        loss_volatility = np.std(recent_losses)
        reward_volatility = np.std(recent_rewards)
        
        # 수렴 판단
        if loss_trend < -0.001 and reward_trend > 0.001:
            status = 'converging'
        elif abs(loss_trend) < 0.001 and abs(reward_trend) < 0.001:
            status = 'converged'
        else:
            status = 'diverging'
            
        return {
            'status': status,
            'loss_trend': loss_trend,
            'reward_trend': reward_trend,
            'loss_volatility': loss_volatility,
            'reward_volatility': reward_volatility
        }
