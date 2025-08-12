"""
메시지 전달 시스템

BioHama 시스템의 모듈 간 정보 교환을 담당합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..core.base.module_interface import ModuleInterface


class MessagePassing(ModuleInterface):
    """
    메시지 전달 시스템
    
    모듈 간 정보 교환을 관리하고 최적화합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        메시지 전달 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 메시지 설정
        self.message_dim = config.get('message_dim', 128)
        self.max_message_length = config.get('max_message_length', 1000)
        self.message_ttl = config.get('message_ttl', 10)  # Time to live
        
        # 메시지 네트워크
        self.message_encoder = nn.Sequential(
            nn.Linear(self.config.get('input_dim', 128), self.message_dim),
            nn.ReLU(),
            nn.Linear(self.message_dim, self.message_dim)
        )
        
        self.message_decoder = nn.Sequential(
            nn.Linear(self.message_dim, self.message_dim),
            nn.ReLU(),
            nn.Linear(self.message_dim, self.config.get('output_dim', 128))
        )
        
        # 메시지 큐와 라우팅 테이블
        self.message_queue = []
        self.routing_table = defaultdict(list)
        self.message_history = []
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지 전달 시스템 순전파
        
        Args:
            inputs: 입력 데이터
            context: 컨텍스트 정보
            
        Returns:
            출력 데이터
        """
        source_module = inputs.get('source_module')
        target_module = inputs.get('target_module')
        message_content = inputs.get('message_content')
        message_type = inputs.get('message_type', 'data')
        
        if source_module and target_module and message_content:
            # 메시지 인코딩
            encoded_message = self._encode_message(message_content)
            
            # 메시지 생성
            message = {
                'id': f"msg_{len(self.message_queue)}_{np.random.randint(1000)}",
                'source': source_module,
                'target': target_module,
                'content': encoded_message,
                'type': message_type,
                'timestamp': len(self.message_queue),
                'ttl': self.message_ttl
            }
            
            # 메시지 전송
            success = self._send_message(message)
            
            return {
                'message_id': message['id'],
                'success': success,
                'encoded_message': encoded_message
            }
        else:
            # 메시지 처리
            processed_messages = self._process_messages()
            
            return {
                'processed_messages': processed_messages,
                'queue_size': len(self.message_queue)
            }
    
    def _encode_message(self, content: Any) -> torch.Tensor:
        """메시지 인코딩"""
        if isinstance(content, torch.Tensor):
            if content.size(0) != self.message_dim:
                content = self.message_encoder(content)
        else:
            # 스칼라나 리스트를 텐서로 변환
            if isinstance(content, (int, float)):
                content = torch.tensor([float(content)])
            elif isinstance(content, list):
                content = torch.tensor(content, dtype=torch.float32)
            else:
                content = torch.randn(self.message_dim)
            
            # 인코딩
            if content.size(0) != self.message_dim:
                content = self.message_encoder(content)
        
        return content
    
    def _send_message(self, message: Dict[str, Any]) -> bool:
        """메시지 전송"""
        # 라우팅 테이블 확인
        if message['target'] in self.routing_table:
            # 메시지 큐에 추가
            self.message_queue.append(message)
            
            # 메시지 히스토리에 추가
            self.message_history.append(message.copy())
            
            # 히스토리 크기 제한
            if len(self.message_history) > self.max_message_length:
                self.message_history = self.message_history[-self.max_message_length:]
            
            return True
        else:
            return False
    
    def _process_messages(self) -> List[Dict[str, Any]]:
        """메시지 처리"""
        processed_messages = []
        messages_to_remove = []
        
        for i, message in enumerate(self.message_queue):
            # TTL 감소
            message['ttl'] -= 1
            
            # TTL이 0이면 제거
            if message['ttl'] <= 0:
                messages_to_remove.append(i)
                continue
            
            # 메시지 디코딩
            decoded_content = self._decode_message(message['content'])
            
            processed_message = {
                'id': message['id'],
                'source': message['source'],
                'target': message['target'],
                'decoded_content': decoded_content,
                'type': message['type'],
                'ttl': message['ttl']
            }
            
            processed_messages.append(processed_message)
        
        # 만료된 메시지 제거
        for i in reversed(messages_to_remove):
            del self.message_queue[i]
        
        return processed_messages
    
    def _decode_message(self, encoded_message: torch.Tensor) -> torch.Tensor:
        """메시지 디코딩"""
        return self.message_decoder(encoded_message)
    
    def add_routing_rule(self, source: str, target: str, priority: float = 1.0) -> None:
        """
        라우팅 규칙 추가
        
        Args:
            source: 소스 모듈
            target: 타겟 모듈
            priority: 우선순위
        """
        self.routing_table[target].append({
            'source': source,
            'priority': priority
        })
    
    def remove_routing_rule(self, source: str, target: str) -> None:
        """
        라우팅 규칙 제거
        
        Args:
            source: 소스 모듈
            target: 타겟 모듈
        """
        if target in self.routing_table:
            self.routing_table[target] = [
                rule for rule in self.routing_table[target]
                if rule['source'] != source
            ]
    
    def get_message_statistics(self) -> Dict[str, Any]:
        """
        메시지 통계 반환
        
        Returns:
            메시지 통계 딕셔너리
        """
        if not self.message_history:
            return {
                'total_messages': 0,
                'queue_size': 0,
                'avg_ttl': 0.0
            }
        
        total_messages = len(self.message_history)
        queue_size = len(self.message_queue)
        
        # 메시지 타입별 통계
        message_types = defaultdict(int)
        for message in self.message_history:
            message_types[message['type']] += 1
        
        return {
            'total_messages': total_messages,
            'queue_size': queue_size,
            'message_types': dict(message_types),
            'routing_table_size': len(self.routing_table)
        }
    
    def clear_message_queue(self) -> None:
        """메시지 큐 초기화"""
        self.message_queue = []
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """상태 업데이트"""
        self.state.update(new_state)
    
    def get_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.state.copy()
    
    def reset(self) -> None:
        """메시지 전달 시스템 초기화"""
        super().reset()
        self.message_queue = []
        self.routing_table.clear()
        self.message_history = []

