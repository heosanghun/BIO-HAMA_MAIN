"""
동적 희소 어텐션 모듈 (M_sparse)

이 모듈은 효율적인 어텐션 메커니즘을 통해 계산 복잡도를 줄이고,
패턴 학습을 통한 동적 어텐션 최적화를 제공합니다.

주요 기능:
- 패턴 기반 어텐션 마스크 생성
- 동적 희소화를 통한 계산 효율성 향상
- O(n²) → O(n) 복잡도 최적화
- 적응형 어텐션 패턴 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import math
import logging

from .base.module_interface import ModuleInterface
from .base.state_interface import StateInterface

logger = logging.getLogger(__name__)


class PatternLearner(nn.Module):
    """어텐션 패턴 학습기"""
    
    def __init__(self, 
                 input_dim: int,
                 pattern_dim: int = 64,
                 num_patterns: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.pattern_dim = pattern_dim
        self.num_patterns = num_patterns
        self.temperature = temperature
        
        # 패턴 학습 네트워크
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.ReLU(),
            nn.Linear(pattern_dim, pattern_dim),
            nn.Tanh()
        )
        
        # 패턴 저장소
        self.pattern_bank = nn.Parameter(
            torch.randn(num_patterns, pattern_dim) * 0.1
        )
        
        # 패턴 가중치 학습
        self.pattern_weights = nn.Parameter(
            torch.ones(num_patterns) / num_patterns
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력에 대한 패턴 매칭 및 가중치 계산
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)
            
        Returns:
            pattern_scores: 패턴 매칭 점수
            pattern_weights: 패턴 가중치
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 인코딩
        encoded = self.pattern_encoder(x)  # (batch_size, seq_len, pattern_dim)
        
        # 패턴 매칭 계산
        pattern_scores = torch.matmul(
            encoded, self.pattern_bank.T
        ) / math.sqrt(self.pattern_dim)  # (batch_size, seq_len, num_patterns)
        
        # 소프트맥스로 패턴 확률 계산
        pattern_probs = F.softmax(pattern_scores / self.temperature, dim=-1)
        
        # 가중치 적용
        weighted_patterns = pattern_probs * self.pattern_weights.unsqueeze(0).unsqueeze(0)
        
        return weighted_patterns, pattern_scores
    
    def update_patterns(self, new_patterns: torch.Tensor):
        """패턴 저장소 업데이트"""
        with torch.no_grad():
            self.pattern_bank.data = new_patterns


class MaskGenerator(nn.Module):
    """희소 어텐션 마스크 생성기"""
    
    def __init__(self,
                 seq_len: int,
                 sparsity_ratio: float = 0.8,
                 local_window: int = 64,
                 num_heads: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.sparsity_ratio = sparsity_ratio
        self.local_window = local_window
        self.num_heads = num_heads
        
        # 마스크 생성 네트워크
        self.mask_predictor = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.ReLU(),
            nn.Linear(seq_len // 2, seq_len),
            nn.Sigmoid()
        )
        
        # 전역 어텐션 토큰 선택기
        self.global_token_selector = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.ReLU(),
            nn.Linear(seq_len // 4, seq_len),
            nn.Sigmoid()
        )
        
    def forward(self, 
                attention_scores: torch.Tensor,
                seq_len: Optional[int] = None) -> torch.Tensor:
        """
        어텐션 마스크 생성
        
        Args:
            attention_scores: 어텐션 점수 (batch_size, seq_len, seq_len) 또는 (batch_size, num_heads, seq_len, seq_len)
            seq_len: 시퀀스 길이 (동적 조정용)
            
        Returns:
            attention_mask: 희소 어텐션 마스크
        """
        if seq_len is None:
            seq_len = self.seq_len
            
        # attention_scores의 차원에 따라 처리
        if attention_scores.dim() == 3:
            # (batch_size, seq_len, seq_len)
            batch_size, seq_len_actual, _ = attention_scores.shape
            num_heads = self.num_heads
            # num_heads 차원 추가
            attention_scores = attention_scores.unsqueeze(1).expand(-1, num_heads, -1, -1)
        else:
            # (batch_size, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len_actual, _ = attention_scores.shape
        
        # 1. 로컬 윈도우 마스크 (대각선 주변)
        local_mask = self._create_local_mask(seq_len_actual, batch_size, num_heads)
        
        # 2. 전역 어텐션 마스크 (중요한 토큰들)
        global_mask = self._create_global_mask(attention_scores, seq_len_actual)
        
        # 3. 랜덤 희소화 마스크
        random_mask = self._create_random_mask(seq_len_actual, batch_size, num_heads)
        
        # 마스크 결합
        combined_mask = (local_mask + global_mask + random_mask) > 0
        
        # 희소성 보장
        final_mask = self._ensure_sparsity(combined_mask, attention_scores)
        
        return final_mask
    
    def _create_local_mask(self, 
                          seq_len: int, 
                          batch_size: int, 
                          num_heads: int) -> torch.Tensor:
        """로컬 윈도우 마스크 생성"""
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2)
            mask[:, :, i, start:end] = 1
            
        return mask
    
    def _create_global_mask(self, 
                           attention_scores: torch.Tensor, 
                           seq_len: int) -> torch.Tensor:
        """전역 어텐션 마스크 생성"""
        batch_size, num_heads, _, _ = attention_scores.shape
        
        # 어텐션 점수의 평균을 사용하여 중요도 계산
        importance = attention_scores.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        # 상위 k개 토큰 선택
        k = max(1, int(seq_len * (1 - self.sparsity_ratio)))
        _, top_indices = torch.topk(importance, k=k, dim=-1)
        
        mask = torch.zeros_like(attention_scores)
        for b in range(batch_size):
            for i in range(seq_len):
                mask[b, :, i, top_indices[b, i]] = 1
                
        return mask
    
    def _create_random_mask(self, 
                           seq_len: int, 
                           batch_size: int, 
                           num_heads: int) -> torch.Tensor:
        """랜덤 희소화 마스크 생성"""
        mask = torch.rand(batch_size, num_heads, seq_len, seq_len) < (1 - self.sparsity_ratio)
        return mask.float()
    
    def _ensure_sparsity(self, 
                         mask: torch.Tensor, 
                         attention_scores: torch.Tensor) -> torch.Tensor:
        """희소성 보장"""
        batch_size, num_heads, seq_len, _ = mask.shape
        
        # 각 행에서 최대 k개만 선택
        k = max(1, int(seq_len * (1 - self.sparsity_ratio)))
        
        # 마스크가 적용된 어텐션 점수
        masked_scores = attention_scores * mask
        
        # 각 행에서 상위 k개 선택
        _, top_indices = torch.topk(masked_scores, k=k, dim=-1)
        
        final_mask = torch.zeros_like(mask)
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    final_mask[b, h, i, top_indices[b, h, i]] = 1
                    
        return final_mask


class AttentionOptimizer(nn.Module):
    """어텐션 최적화기"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        # 선형 변환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 드롭아웃
        self.dropout_layer = nn.Dropout(dropout)
        
        # 스케일링 팩터
        self.scale = math.sqrt(self.d_k)
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        최적화된 어텐션 계산
        
        Args:
            query: 쿼리 텐서
            key: 키 텐서
            value: 값 텐서
            mask: 어텐션 마스크
            
        Returns:
            output: 어텐션 출력
        """
        batch_size = query.size(0)
        
        # 선형 변환
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Flash Attention 사용 (가능한 경우)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=self.dropout)
        else:
            # 표준 어텐션 계산
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout_layer(attention_weights)
            
            output = torch.matmul(attention_weights, V)
        
        # 출력 변환
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output


class SparseAttentionModule(ModuleInterface, nn.Module):
    """동적 희소 어텐션 모듈"""
    
    def __init__(self, 
                 config: Dict[str, Any]):
        ModuleInterface.__init__(self, config)
        nn.Module.__init__(self)
        
        # 설정 파라미터
        self.d_model = config.get('d_model', 512)
        self.num_heads = config.get('num_heads', 8)
        self.seq_len = config.get('seq_len', 1024)
        self.sparsity_ratio = config.get('sparsity_ratio', 0.8)
        self.local_window = config.get('local_window', 64)
        self.pattern_dim = config.get('pattern_dim', 64)
        self.num_patterns = config.get('num_patterns', 8)
        self.use_flash_attention = config.get('use_flash_attention', True)
        
        # 모듈 구성 요소
        self.pattern_learner = PatternLearner(
            input_dim=self.d_model,
            pattern_dim=self.pattern_dim,
            num_patterns=self.num_patterns
        )
        
        self.mask_generator = MaskGenerator(
            seq_len=self.seq_len,
            sparsity_ratio=self.sparsity_ratio,
            local_window=self.local_window,
            num_heads=self.num_heads
        )
        
        self.attention_optimizer = AttentionOptimizer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_flash_attention=self.use_flash_attention
        )
        
        # 통계 추적
        self.attention_stats = {
            'total_tokens': 0,
            'sparse_tokens': 0,
            'pattern_matches': 0,
            'computation_savings': 0.0
        }
        
        # 활성화 상태
        self.is_active = True
        
    def forward(self, 
                inputs: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        희소 어텐션 처리
        
        Args:
            inputs: 입력 데이터
                - 'query': 쿼리 텐서
                - 'key': 키 텐서  
                - 'value': 값 텐서
                - 'seq_len': 시퀀스 길이 (선택사항)
                
        Returns:
            outputs: 처리 결과
                - 'attention_output': 어텐션 출력
                - 'attention_mask': 사용된 마스크
                - 'pattern_weights': 패턴 가중치
                - 'sparsity_ratio': 실제 희소성 비율
        """
        if not self.is_active:
            return inputs
            
        query = inputs.get('query')
        key = inputs.get('key', query)
        value = inputs.get('value', query)
        seq_len = inputs.get('seq_len', self.seq_len)
        
        if query is None:
            logger.warning("SparseAttention: query가 제공되지 않았습니다.")
            return inputs
            
        batch_size, seq_len_actual, _ = query.shape
        
        # 1. 패턴 학습 및 매칭
        pattern_weights, pattern_scores = self.pattern_learner(query)
        
        # 2. 어텐션 점수 계산 (전체)
        Q = self.attention_optimizer.w_q(query)
        K = self.attention_optimizer.w_k(key)
        
        # 3. 희소 마스크 생성
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_optimizer.d_k)
        attention_mask = self.mask_generator(attention_scores, seq_len_actual)
        
        # 4. 최적화된 어텐션 계산
        attention_output = self.attention_optimizer(query, key, value, attention_mask)
        
        # 5. 통계 업데이트
        self._update_stats(attention_mask, pattern_scores, seq_len_actual)
        
        # 6. 상태 업데이트
        self.update_state({
            'last_attention_mask': attention_mask.detach(),
            'last_pattern_weights': pattern_weights.detach(),
            'last_sparsity_ratio': 1 - attention_mask.float().mean().item()
        })
        
        return {
            'attention_output': attention_output,
            'attention_mask': attention_mask,
            'pattern_weights': pattern_weights,
            'sparsity_ratio': 1 - attention_mask.float().mean().item(),
            'pattern_scores': pattern_scores,
            'computation_savings': self.attention_stats['computation_savings']
        }
    
    def _update_stats(self, 
                     attention_mask: torch.Tensor,
                     pattern_scores: torch.Tensor,
                     seq_len: int):
        """통계 업데이트"""
        total_tokens = attention_mask.numel()
        sparse_tokens = (attention_mask == 0).sum().item()
        
        self.attention_stats['total_tokens'] += total_tokens
        self.attention_stats['sparse_tokens'] += sparse_tokens
        self.attention_stats['pattern_matches'] += pattern_scores.max(dim=-1)[0].sum().item()
        self.attention_stats['computation_savings'] = (
            self.attention_stats['sparse_tokens'] / 
            max(1, self.attention_stats['total_tokens'])
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
            'module_type': 'sparse_attention',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'sparsity_ratio': self.sparsity_ratio,
            'is_active': self.is_active
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """복잡도 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': 'sparse_attention',
            'computation_savings': self.attention_stats['computation_savings']
        }
    
    def activate(self):
        """모듈 활성화"""
        self.is_active = True
        logger.info(f"{self.name} 모듈이 활성화되었습니다.")
    
    def deactivate(self):
        """모듈 비활성화"""
        self.is_active = False
        logger.info(f"{self.name} 모듈이 비활성화되었습니다.")
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """어텐션 통계 반환"""
        return self.attention_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.attention_stats = {
            'total_tokens': 0,
            'sparse_tokens': 0,
            'pattern_matches': 0,
            'computation_savings': 0.0
        }
    
    def update_patterns(self, new_patterns: torch.Tensor):
        """패턴 업데이트"""
        self.pattern_learner.update_patterns(new_patterns)
        logger.info(f"{self.name}: 패턴이 업데이트되었습니다.")
    
    def set_sparsity_ratio(self, ratio: float):
        """희소성 비율 설정"""
        self.sparsity_ratio = max(0.0, min(1.0, ratio))
        logger.info(f"{self.name}: 희소성 비율이 {self.sparsity_ratio:.2f}로 설정되었습니다.")


class SparseAttentionState(StateInterface):
    """희소 어텐션 상태 관리"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attention_history = []
        self.pattern_history = []
        self.sparsity_history = []
        self.performance_metrics = {}
        
    def update_state(self, new_state: Dict[str, Any]):
        """상태 업데이트"""
        timestamp = len(self.attention_history)
        
        self.attention_history.append({
            'timestamp': timestamp,
            'attention_mask': new_state.get('attention_mask'),
            'pattern_weights': new_state.get('pattern_weights'),
            'sparsity_ratio': new_state.get('sparsity_ratio', 0.0)
        })
        
        # 성능 메트릭 업데이트
        if 'computation_savings' in new_state:
            self.performance_metrics['computation_savings'] = new_state['computation_savings']
    
    def get_current_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        if not self.attention_history:
            return {}
            
        latest = self.attention_history[-1]
        return {
            'current_attention_mask': latest['attention_mask'],
            'current_pattern_weights': latest['pattern_weights'],
            'current_sparsity_ratio': latest['sparsity_ratio'],
            'performance_metrics': self.performance_metrics
        }
    
    def get_state_at_time(self, timestamp: int) -> Dict[str, Any]:
        """특정 시점의 상태 반환"""
        if 0 <= timestamp < len(self.attention_history):
            return self.attention_history[timestamp]
        return {}
    
    def get_attention_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """어텐션 트렌드 분석"""
        if len(self.attention_history) < window_size:
            return {}
            
        recent_history = self.attention_history[-window_size:]
        
        sparsity_ratios = [h['sparsity_ratio'] for h in recent_history]
        
        return {
            'avg_sparsity': np.mean(sparsity_ratios),
            'sparsity_trend': np.polyfit(range(len(sparsity_ratios)), sparsity_ratios, 1)[0],
            'sparsity_volatility': np.std(sparsity_ratios)
        }
