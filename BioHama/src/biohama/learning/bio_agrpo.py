"""
Bio-A-GRPO: 바이오-인스파이어드 적응형 정책 최적화

뇌과학적 기반의 강화학습 알고리즘으로, 신경전달물질 시스템을
모방하여 적응적이고 효율적인 학습을 수행합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import random

from ..core.base.module_interface import ModuleInterface


class BioAGRPO(ModuleInterface):
    """
    Bio-A-GRPO 알고리즘
    
    바이오-인스파이어드 적응형 정책 최적화 알고리즘으로,
    뇌의 신경전달물질 시스템을 모방하여 학습을 수행합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Bio-A-GRPO 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 알고리즘 설정
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 신경전달물질 설정
        self.dopamine_decay = config.get('dopamine_decay', 0.95)
        self.serotonin_modulation = config.get('serotonin_modulation', 0.1)
        self.norepinephrine_boost = config.get('norepinephrine_boost', 0.2)
        
        # 경험 리플레이
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
        # 신경망 구성 요소들
        self._build_networks()
        
        # 학습 상태
        self.training_step = 0
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        
    def _build_networks(self):
        """신경망 구성 요소들을 구축합니다."""
        
        # 정책 네트워크 (Actor)
        self.policy_network = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.get('action_dim', 64))
        )
        
        # 가치 네트워크 (Critic)
        self.value_network = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 신경전달물질 네트워크
        self.neurotransmitter_network = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 도파민, 세로토닌, 노르에피네프린
        )
        
        # 옵티마이저
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=self.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), 
            lr=self.learning_rate
        )
        self.neurotransmitter_optimizer = torch.optim.Adam(
            self.neurotransmitter_network.parameters(), 
            lr=self.learning_rate
        )
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Bio-A-GRPO 순전파
        
        Args:
            inputs: 입력 데이터
            context: 컨텍스트 정보
            
        Returns:
            출력 데이터
        """
        state = inputs.get('state')
        if state is None:
            return {'error': 'State not provided'}
        
        # 상태를 텐서로 변환
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # 정책과 가치 계산
        policy_logits = self.policy_network(state)
        value = self.value_network(state)
        
        # 신경전달물질 수준 계산
        neurotransmitter_levels = self.neurotransmitter_network(state)
        dopamine, serotonin, norepinephrine = torch.split(neurotransmitter_levels, 1, dim=-1)
        
        # 행동 확률 계산
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # 탐색 vs 활용 결정
        if context and context.get('training', False):
            # 훈련 중: 탐색
            action = torch.multinomial(action_probs, 1)
        else:
            # 추론 중: 활용
            action = torch.argmax(action_probs, dim=-1, keepdim=True)
        
        return {
            'action': action,
            'action_probs': action_probs,
            'value': value,
            'dopamine': dopamine,
            'serotonin': serotonin,
            'norepinephrine': norepinephrine,
            'policy_logits': policy_logits
        }
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """상태 업데이트"""
        self.state.update(new_state)
    
    def get_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.state.copy()
    
    def store_experience(self, experience: Dict[str, Any]) -> None:
        """
        경험을 리플레이 버퍼에 저장
        
        Args:
            experience: 경험 데이터
        """
        self.replay_buffer.append(experience)
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool]) -> List[float]:
        """
        GAE(Generalized Advantage Estimation) 계산
        
        Args:
            rewards: 보상 시퀀스
            values: 가치 시퀀스
            dones: 종료 플래그 시퀀스
            
        Returns:
            어드밴티지 시퀀스
        """
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                     advantages: torch.Tensor, old_log_probs: torch.Tensor,
                     returns: torch.Tensor) -> Dict[str, float]:
        """
        정책 업데이트
        
        Args:
            states: 상태 배치
            actions: 행동 배치
            advantages: 어드밴티지 배치
            old_log_probs: 이전 로그 확률 배치
            returns: 반환값 배치
            
        Returns:
            손실 정보
        """
        # 현재 정책의 로그 확률 계산
        policy_logits = self.policy_network(states)
        action_probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # 선택된 행동의 로그 확률
        action_log_probs = log_probs.gather(1, actions)
        
        # 비율 계산
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # 클리핑된 목적 함수
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 엔트로피 손실
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 가치 손실
        value_pred = self.value_network(states)
        value_loss = F.mse_loss(value_pred, returns)
        
        # 총 손실
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # 옵티마이저 스텝
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def update_neurotransmitters(self, states: torch.Tensor, 
                               rewards: torch.Tensor) -> Dict[str, float]:
        """
        신경전달물질 네트워크 업데이트
        
        Args:
            states: 상태 배치
            rewards: 보상 배치
            
        Returns:
            손실 정보
        """
        # 신경전달물질 수준 예측
        neurotransmitter_pred = self.neurotransmitter_network(states)
        
        # 목표 신경전달물질 수준 계산 (보상 기반)
        target_dopamine = torch.clamp(rewards, 0, 1)  # 긍정적 보상
        target_serotonin = torch.ones_like(rewards) * 0.5  # 안정성
        target_norepinephrine = torch.abs(rewards)  # 각성
        
        target_levels = torch.cat([target_dopamine, target_serotonin, target_norepinephrine], dim=-1)
        
        # 손실 계산
        loss = F.mse_loss(neurotransmitter_pred, target_levels)
        
        # 옵티마이저 스텝
        self.neurotransmitter_optimizer.zero_grad()
        loss.backward()
        self.neurotransmitter_optimizer.step()
        
        return {'neurotransmitter_loss': loss.item()}
    
    def train_step(self) -> Dict[str, float]:
        """
        한 스텝의 훈련 수행
        
        Returns:
            훈련 손실 정보
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'error': 'Insufficient experience'}
        
        # 배치 샘플링
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 배치 데이터 준비
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool)
        
        # 가치 계산
        with torch.no_grad():
            values = self.value_network(states).squeeze(-1)
        
        # 어드밴티지 계산
        advantages = self.compute_advantages(
            rewards.tolist(), values.tolist(), dones.tolist()
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 반환값 계산
        returns = advantages + values
        
        # 이전 로그 확률 (간단한 구현)
        with torch.no_grad():
            old_logits = self.policy_network(states)
            old_log_probs = F.log_softmax(old_logits, dim=-1).gather(1, actions)
        
        # 정책 업데이트
        policy_losses = self.update_policy(states, actions, advantages, old_log_probs, returns)
        
        # 신경전달물질 업데이트
        neurotransmitter_losses = self.update_neurotransmitters(states, rewards)
        
        # 손실 기록
        self.policy_losses.append(policy_losses['policy_loss'])
        self.value_losses.append(policy_losses['value_loss'])
        
        self.training_step += 1
        
        return {**policy_losses, **neurotransmitter_losses}
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        훈련 통계 반환
        
        Returns:
            훈련 통계 딕셔너리
        """
        return {
            'training_step': self.training_step,
            'replay_buffer_size': len(self.replay_buffer),
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'episode_count': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        체크포인트 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        checkpoint = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'neurotransmitter_network_state_dict': self.neurotransmitter_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'neurotransmitter_optimizer_state_dict': self.neurotransmitter_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        체크포인트 로드
        
        Args:
            filepath: 로드할 파일 경로
        """
        checkpoint = torch.load(filepath)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.neurotransmitter_network.load_state_dict(checkpoint['neurotransmitter_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.neurotransmitter_optimizer.load_state_dict(checkpoint['neurotransmitter_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
    
    def reset(self) -> None:
        """Bio-A-GRPO 초기화"""
        super().reset()
        self.replay_buffer.clear()
        self.training_step = 0
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []

