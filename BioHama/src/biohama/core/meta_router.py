"""
메타-라우터 구현

계층적 의사결정과 라우팅을 담당하는 메타-라우터의 핵심 구현입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base.router_interface import RouterInterface


class MetaRouter(RouterInterface):
    """
    계층적 메타-라우터
    
    입력을 분석하여 적절한 인지 모듈로 라우팅하는
    고차 인지 의사결정 시스템입니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        메타-라우터 초기화
        
        Args:
            config: 라우터 설정
        """
        super().__init__(config)
        
        # 아키텍처 설정
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # 라우팅 설정
        self.routing_temperature = config.get('routing_temperature', 1.0)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_history_length = config.get('max_history_length', 1000)
        
        # 신경망 구성 요소들
        self._build_networks()
        
        # 라우팅 상태
        self.module_embeddings = {}
        self.routing_weights = {}
        self.attention_cache = {}
        
    def _build_networks(self):
        """신경망 구성 요소들을 구축합니다."""
        
        # 입력 인코더
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 멀티헤드 어텐션
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 피드포워드 네트워크
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        # 라우팅 헤드
        self.routing_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # 컨텍스트 인코더
        self.context_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 레이어 정규화
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
    def route(self, inputs: Dict[str, Any], available_modules: List[Any], 
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        입력을 적절한 모듈로 라우팅
        
        Args:
            inputs: 입력 데이터
            available_modules: 사용 가능한 모듈 목록
            context: 컨텍스트 정보
            
        Returns:
            라우팅 결과
        """
        # 입력 전처리
        processed_inputs = self._preprocess_inputs(inputs)
        
        # 컨텍스트 처리
        context_features = self._process_context(context) if context else None
        
        # 모듈 임베딩 생성
        module_features = self._get_module_features(available_modules)
        
        # 라우팅 점수 계산
        routing_scores = self._compute_routing_scores(
            processed_inputs, module_features, context_features
        )
        
        # 라우팅 결정
        routing_decision = self._make_routing_decision(
            routing_scores, available_modules
        )
        
        # 결과 기록
        self._record_routing_decision(inputs, routing_decision, routing_scores)
        
        return routing_decision
    
    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """입력 데이터를 전처리합니다."""
        
        # 다양한 입력 타입 처리
        if 'text' in inputs:
            # 텍스트 입력 처리
            text_features = self._encode_text(inputs['text'])
            return text_features
        elif 'image' in inputs:
            # 이미지 입력 처리
            image_features = self._encode_image(inputs['image'])
            return image_features
        elif 'audio' in inputs:
            # 오디오 입력 처리
            audio_features = self._encode_audio(inputs['audio'])
            return audio_features
        elif 'multimodal' in inputs:
            # 다중모달 입력 처리
            multimodal_features = self._encode_multimodal(inputs['multimodal'])
            return multimodal_features
        else:
            # 기본 입력 처리
            return self._encode_generic(inputs)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """텍스트를 인코딩합니다."""
        # 간단한 임베딩 (실제로는 더 정교한 토크나이저 사용)
        import hashlib
        hash_value = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        embedding = torch.randn(self.input_dim)
        embedding[0] = hash_value % 1000 / 1000.0  # 해시 기반 특징
        return embedding.unsqueeze(0)
    
    def _encode_image(self, image: Any) -> torch.Tensor:
        """이미지를 인코딩합니다."""
        # 간단한 이미지 특징 추출 (실제로는 CNN 사용)
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            # 평균 풀링으로 특징 추출
            features = F.adaptive_avg_pool2d(image, (1, 1)).flatten(1)
            if features.size(1) != self.input_dim:
                features = F.linear(features, 
                                  torch.randn(self.input_dim, features.size(1)))
            return features
        else:
            return torch.randn(1, self.input_dim)
    
    def _encode_audio(self, audio: Any) -> torch.Tensor:
        """오디오를 인코딩합니다."""
        # 간단한 오디오 특징 추출
        return torch.randn(1, self.input_dim)
    
    def _encode_multimodal(self, multimodal_data: Dict[str, Any]) -> torch.Tensor:
        """다중모달 데이터를 인코딩합니다."""
        features = []
        for modality, data in multimodal_data.items():
            if modality == 'text':
                features.append(self._encode_text(data))
            elif modality == 'image':
                features.append(self._encode_image(data))
            elif modality == 'audio':
                features.append(self._encode_audio(data))
        
        # 특징 융합
        if features:
            combined = torch.cat(features, dim=1)
            if combined.size(1) != self.input_dim:
                combined = F.linear(combined, 
                                  torch.randn(self.input_dim, combined.size(1)))
            return combined
        else:
            return torch.randn(1, self.input_dim)
    
    def _encode_generic(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """일반적인 입력을 인코딩합니다."""
        # 딕셔너리를 평면화하여 특징 벡터 생성
        flat_features = []
        for key, value in inputs.items():
            if isinstance(value, (int, float)):
                flat_features.append(float(value))
            elif isinstance(value, str):
                # 문자열을 해시로 변환
                hash_val = hash(value) % 1000 / 1000.0
                flat_features.append(hash_val)
        
        # 특징 벡터 생성
        if flat_features:
            feature_tensor = torch.tensor(flat_features, dtype=torch.float32)
            if len(feature_tensor) != self.input_dim:
                # 패딩 또는 자르기
                if len(feature_tensor) < self.input_dim:
                    padding = torch.zeros(self.input_dim - len(feature_tensor))
                    feature_tensor = torch.cat([feature_tensor, padding])
                else:
                    feature_tensor = feature_tensor[:self.input_dim]
        else:
            feature_tensor = torch.randn(self.input_dim)
        
        return feature_tensor.unsqueeze(0)
    
    def _process_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """컨텍스트 정보를 처리합니다."""
        if not context:
            return torch.zeros(1, self.hidden_dim)
        
        # 컨텍스트 특징 추출
        context_features = []
        for key, value in context.items():
            if isinstance(value, torch.Tensor):
                context_features.append(value)
            else:
                # 스칼라 값을 텐서로 변환
                context_features.append(torch.tensor([float(value)]))
        
        if context_features:
            context_tensor = torch.cat(context_features).unsqueeze(0)
            if context_tensor.size(1) != self.hidden_dim:
                context_tensor = F.linear(context_tensor, 
                                        torch.randn(self.hidden_dim, context_tensor.size(1)))
        else:
            context_tensor = torch.zeros(1, self.hidden_dim)
        
        return context_tensor
    
    def _get_module_features(self, modules: List[Any]) -> torch.Tensor:
        """모듈들의 특징을 추출합니다."""
        module_features = []
        
        for i, module in enumerate(modules):
            # 모듈 식별자 생성 (module_id가 없으면 이름 사용)
            module_id = getattr(module, 'module_id', None) or getattr(module, 'name', f'module_{i}')
            
            if module_id in self.module_embeddings:
                features = self.module_embeddings[module_id]
            else:
                # 모듈 특징 생성
                features = self._generate_module_embedding(module)
                self.module_embeddings[module_id] = features
            
            module_features.append(features)
        
        if module_features:
            return torch.stack(module_features)
        else:
            return torch.zeros(0, self.hidden_dim)
    
    def _generate_module_embedding(self, module: Any) -> torch.Tensor:
        """모듈의 임베딩을 생성합니다."""
        # 모듈의 메타데이터를 기반으로 임베딩 생성
        metadata = module.get_metadata()
        complexity = module.get_complexity()
        
        # 특징 벡터 구성
        features = []
        
        # 복잡도 특징
        features.extend([
            complexity.get('total_parameters', 0) / 1e6,  # 백만 단위
            complexity.get('trainable_parameters', 0) / 1e6,
            float(complexity.get('module_type', 'unknown') != 'unknown')
        ])
        
        # 메타데이터 특징
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(hash(value) % 1000 / 1000.0)
        
        # 특징 벡터 정규화
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        if len(feature_tensor) != self.hidden_dim:
            if len(feature_tensor) < self.hidden_dim:
                padding = torch.zeros(self.hidden_dim - len(feature_tensor))
                feature_tensor = torch.cat([feature_tensor, padding])
            else:
                feature_tensor = feature_tensor[:self.hidden_dim]
        
        return feature_tensor
    
    def _compute_routing_scores(self, inputs: torch.Tensor, 
                               module_features: torch.Tensor,
                               context_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """라우팅 점수를 계산합니다."""
        
        # 입력 인코딩
        encoded_inputs = self.input_encoder(inputs)
        
        # 어텐션 처리
        if context_features is not None:
            # 컨텍스트와 결합
            combined_features = torch.cat([encoded_inputs, context_features], dim=1)
            attended_features, _ = self.attention(
                combined_features, combined_features, combined_features
            )
        else:
            attended_features, _ = self.attention(
                encoded_inputs, encoded_inputs, encoded_inputs
            )
        
        # 레이어 정규화
        attended_features = self.layer_norm1(attended_features + encoded_inputs)
        
        # 피드포워드 처리
        ff_output = self.feed_forward(attended_features)
        ff_output = self.layer_norm2(ff_output + attended_features)
        
        # 모듈과의 호환성 계산
        if module_features.size(0) > 0:
            # 모듈 특징과의 유사도 계산
            compatibility_scores = torch.matmul(ff_output, module_features.transpose(0, 1))
            
            # 라우팅 헤드로 최종 점수 계산
            routing_scores = self.routing_head(ff_output).squeeze(-1)
            
            # 호환성과 라우팅 점수 결합
            final_scores = compatibility_scores + routing_scores.unsqueeze(1)
        else:
            final_scores = torch.zeros(1, 0)
        
        return final_scores
    
    def _make_routing_decision(self, routing_scores: torch.Tensor, 
                              available_modules: List[Any]) -> Dict[str, Any]:
        """라우팅 결정을 내립니다."""
        
        if routing_scores.size(1) == 0:
            return {
                'selected_module': None,
                'confidence': 0.0,
                'exploration': False,
                'scores': []
            }
        
        # 소프트맥스로 확률 계산
        probabilities = F.softmax(routing_scores / self.routing_temperature, dim=1)
        
        # 탐색 vs 활용 결정
        if torch.rand(1) < self.exploration_rate:
            # 탐색: 랜덤 선택
            selected_idx = torch.randint(0, len(available_modules), (1,))
            exploration = True
        else:
            # 활용: 최고 점수 선택
            selected_idx = torch.argmax(probabilities, dim=1)
            exploration = False
        
        selected_module = available_modules[selected_idx.item()]
        confidence = probabilities[0, selected_idx.item()].item()
        
        return {
            'selected_module': selected_module,
            'confidence': confidence,
            'exploration': exploration,
            'scores': probabilities[0].tolist(),
            'module_id': selected_module.module_id if hasattr(selected_module, 'module_id') else None
        }
    
    def _record_routing_decision(self, inputs: Dict[str, Any], 
                                decision: Dict[str, Any], 
                                scores: torch.Tensor) -> None:
        """라우팅 결정을 기록합니다."""
        
        record = {
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None,
            'inputs': {k: str(v)[:100] for k, v in inputs.items()},  # 입력 요약
            'decision': decision,
            'scores': scores.tolist() if scores.numel() > 0 else [],
            'exploration': decision.get('exploration', False)
        }
        
        self.routing_history.append(record)
        
        # 히스토리 크기 제한
        if len(self.routing_history) > self.max_history_length:
            self.routing_history = self.routing_history[-self.max_history_length:]
    
    def update_routing_weights(self, feedback: Dict[str, Any]) -> None:
        """라우팅 가중치를 업데이트합니다."""
        
        # 피드백에서 성능 정보 추출
        performance = feedback.get('performance', {})
        module_id = feedback.get('module_id')
        
        if module_id and performance:
            # 모듈별 성능 기록
            if module_id not in self.routing_weights:
                self.routing_weights[module_id] = {
                    'success_count': 0,
                    'total_count': 0,
                    'avg_performance': 0.0
                }
            
            weight_info = self.routing_weights[module_id]
            weight_info['total_count'] += 1
            
            # 성능 점수 계산
            performance_score = performance.get('score', 0.0)
            if performance_score > 0.5:  # 성공 임계값
                weight_info['success_count'] += 1
            
            # 평균 성능 업데이트
            current_avg = weight_info['avg_performance']
            total_count = weight_info['total_count']
            weight_info['avg_performance'] = (
                (current_avg * (total_count - 1) + performance_score) / total_count
            )
    
    def get_routing_confidence(self, module_id: str) -> float:
        """특정 모듈에 대한 라우팅 신뢰도를 반환합니다."""
        
        if module_id in self.routing_weights:
            weight_info = self.routing_weights[module_id]
            if weight_info['total_count'] > 0:
                success_rate = weight_info['success_count'] / weight_info['total_count']
                avg_performance = weight_info['avg_performance']
                return (success_rate + avg_performance) / 2
        
        return 0.5  # 기본 신뢰도
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """라우팅 통계를 반환합니다."""
        
        if not self.routing_history:
            return {}
        
        total_decisions = len(self.routing_history)
        exploration_count = sum(1 for record in self.routing_history 
                              if record.get('exploration', False))
        
        # 모듈별 사용 통계
        module_usage = {}
        for record in self.routing_history:
            module_id = record['decision'].get('module_id')
            if module_id:
                module_usage[module_id] = module_usage.get(module_id, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'exploration_rate': exploration_count / total_decisions if total_decisions > 0 else 0,
            'module_usage': module_usage,
            'routing_weights': self.routing_weights
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            'name': self.name,
            'routing_strategy': self.routing_strategy,
            'device': self.device,
            'module_type': 'router'
        }
    
    def get_complexity(self) -> Dict[str, Any]:
        """모듈 복잡도를 반환합니다."""
        total_params = sum(p.numel() for p in self.parameters()) if hasattr(self, 'parameters') else 0
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) if hasattr(self, 'parameters') else 0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_type': 'router'
        }
