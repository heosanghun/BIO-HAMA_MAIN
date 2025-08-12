"""
BioHama 시스템 메인 클래스

모든 BioHama 구성 요소들을 통합하여 완전한 인공지능 시스템을 구성합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

from .core.meta_router import MetaRouter
from .core.cognitive_state import CognitiveState
from .core.working_memory import WorkingMemory
from .core.decision_engine import DecisionEngine
from .core.attention_control import AttentionControl
from .core.sparse_attention import SparseAttentionModule
from .core.termination_module import TerminationModule
from .communication.message_passing import MessagePassing
from .learning.bio_agrpo import BioAGRPO
from .utils.config import get_config


class BioHamaSystem:
    """
    BioHama 메인 시스템
    
    모든 BioHama 구성 요소들을 통합하여 완전한 인공지능 시스템을 구성합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        BioHama 시스템 초기화
        
        Args:
            config: 시스템 설정 딕셔너리 (None이면 기본 설정 사용)
        """
        # 설정 로드
        if config is None:
            config = get_config()
        
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 시스템 상태
        self.is_initialized = False
        self.is_running = False
        self.system_start_time = None
        
        # 구성 요소들 초기화
        self._initialize_components()
        
        # 시스템 통합
        self._integrate_system()
        
        self.is_initialized = True
        self.logger.info("BioHama 시스템이 성공적으로 초기화되었습니다.")
        
    def _initialize_components(self):
        """시스템 구성 요소들을 초기화합니다."""
        
        # 메타 라우터
        router_config = self.config.get('meta_router', {})
        router_config['device'] = self.device
        self.meta_router = MetaRouter(router_config)
        
        # 인지 상태 관리자
        cognitive_config = self.config.get('cognitive_state', {})
        cognitive_config['device'] = self.device
        self.cognitive_state = CognitiveState(cognitive_config)
        
        # 작업 메모리
        memory_config = self.config.get('working_memory', {})
        memory_config['device'] = self.device
        self.working_memory = WorkingMemory(memory_config)
        
        # 의사결정 엔진
        decision_config = self.config.get('decision_engine', {})
        decision_config['device'] = self.device
        self.decision_engine = DecisionEngine(decision_config)
        
        # 주의 제어
        attention_config = self.config.get('attention_control', {})
        attention_config['device'] = self.device
        self.attention_control = AttentionControl(attention_config)
        
        # 희소 어텐션 모듈
        sparse_config = self.config.get('sparse_attention', {})
        sparse_config['device'] = self.device
        self.sparse_attention = SparseAttentionModule(sparse_config)
        
        # 종료 모듈
        termination_config = self.config.get('termination_module', {})
        termination_config['device'] = self.device
        self.termination_module = TerminationModule(termination_config)
        
        # 메시지 전달 시스템
        message_config = self.config.get('message_passing', {})
        message_config['device'] = self.device
        self.message_passing = MessagePassing(message_config)
        
        # Bio-A-GRPO 학습 알고리즘
        learning_config = self.config.get('bio_agrpo', {})
        learning_config['device'] = self.device
        self.bio_agrpo = BioAGRPO(learning_config)
        
        # 모듈 레지스트리
        self.modules = {}
        self.module_registry = {}
        
    def _integrate_system(self):
        """시스템 구성 요소들을 통합합니다."""
        
        # 상태 관찰자 등록
        self.cognitive_state.add_state_observer(self.meta_router)
        self.cognitive_state.add_state_observer(self.attention_control)
        self.cognitive_state.add_state_observer(self.decision_engine)
        
        # 메시지 라우팅 설정
        self._setup_message_routing()
        
        # 모듈 간 연결 설정
        self._setup_module_connections()
        
    def _setup_message_routing(self):
        """메시지 라우팅을 설정합니다."""
        
        # 기본 라우팅 규칙 설정
        routing_rules = [
            ('cognitive_state', 'meta_router'),
            ('meta_router', 'decision_engine'),
            ('decision_engine', 'attention_control'),
            ('attention_control', 'working_memory'),
            ('working_memory', 'cognitive_state'),
            ('bio_agrpo', 'meta_router'),
            ('meta_router', 'bio_agrpo')
        ]
        
        for source, target in routing_rules:
            self.message_passing.add_routing_rule(source, target)
            
    def _setup_module_connections(self):
        """모듈 간 연결을 설정합니다."""
        
        # 모듈들을 레지스트리에 등록
        self.modules = {
            'meta_router': self.meta_router,
            'cognitive_state': self.cognitive_state,
            'working_memory': self.working_memory,
            'decision_engine': self.decision_engine,
            'attention_control': self.attention_control,
            'sparse_attention': self.sparse_attention,
            'termination_module': self.termination_module,
            'message_passing': self.message_passing,
            'bio_agrpo': self.bio_agrpo
        }
        
        # 모듈 메타데이터 설정
        for module_id, module in self.modules.items():
            if hasattr(module, 'set_metadata'):
                module.set_metadata('module_id', module_id)
                module.set_metadata('system_component', True)
            self.module_registry[module_id] = {
                'module': module,
                'status': 'active',
                'last_activity': datetime.now().isoformat()
            }
            
    def process_input(self, inputs: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        입력을 처리하여 응답을 생성합니다.
        
        Args:
            inputs: 입력 데이터
            context: 컨텍스트 정보
            
        Returns:
            처리 결과
        """
        if not self.is_initialized:
            raise RuntimeError("BioHama 시스템이 초기화되지 않았습니다.")
            
        self.logger.info(f"입력 처리 시작: {list(inputs.keys())}")
        
        # 1. 인지 상태 업데이트
        cognitive_update = {
            'input_received': True,
            'input_type': list(inputs.keys()),
            'timestamp': datetime.now().isoformat()
        }
        self.cognitive_state.update_state(cognitive_update)
        
        # 2. 메타 라우터를 통한 라우팅
        available_modules = list(self.modules.values())
        routing_result = self.meta_router.route(inputs, available_modules, context)
        
        # 3. 선택된 모듈에서 처리
        selected_module = routing_result.get('selected_module')
        if selected_module:
            module_output = selected_module.forward(inputs, context)
            
            # 4. 작업 메모리에 결과 저장
            if 'output' in module_output:
                memory_id = self.working_memory.store(
                    torch.tensor(module_output['output']),
                    priority=routing_result.get('confidence', 0.5)
                )
                module_output['memory_id'] = memory_id
            
            # 5. 주의 상태 업데이트
            if 'attention_target' in module_output:
                self.attention_control.update_attention_focus(
                    torch.tensor(module_output['attention_target']),
                    torch.tensor(module_output.get('salience', [1.0]))
                )
            
            # 6. 학습 데이터 수집
            if 'learning_data' in module_output:
                self.bio_agrpo.store_experience(module_output['learning_data'])
            
            # 7. 메시지 전달
            self._process_messages(module_output)
            
            return {
                'output': module_output,
                'routing_confidence': routing_result.get('confidence', 0.0),
                'selected_module': selected_module.name,
                'cognitive_state': self.cognitive_state.get_cognitive_state_summary(),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'error': '적절한 처리 모듈을 찾을 수 없습니다.',
                'routing_confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
    def _process_messages(self, module_output: Dict[str, Any]):
        """모듈 출력을 기반으로 메시지를 처리합니다."""
        
        if 'messages' in module_output:
            for message in module_output['messages']:
                self.message_passing.forward({
                    'source_module': message.get('source'),
                    'target_module': message.get('target'),
                    'message_content': message.get('content'),
                    'message_type': message.get('type', 'data')
                })
                
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        시스템을 훈련합니다.
        
        Args:
            training_data: 훈련 데이터
            
        Returns:
            훈련 결과
        """
        if not self.is_initialized:
            raise RuntimeError("BioHama 시스템이 초기화되지 않았습니다.")
            
        self.logger.info(f"훈련 시작: {len(training_data)} 개의 샘플")
        
        training_results = []
        
        for i, data in enumerate(training_data):
            # 입력 처리
            result = self.process_input(data['input'], data.get('context'))
            
            # 피드백 처리
            if 'feedback' in data:
                self._process_feedback(data['feedback'], result)
            
            # 학습 단계 수행
            if i % 10 == 0:  # 10개 샘플마다 학습
                learning_result = self.bio_agrpo.train_step()
                training_results.append(learning_result)
                
        return {
            'training_samples': len(training_data),
            'learning_results': training_results,
            'final_statistics': self.get_system_statistics()
        }
        
    def _process_feedback(self, feedback: Dict[str, Any], result: Dict[str, Any]):
        """피드백을 처리하여 시스템을 업데이트합니다."""
        
        # 메타 라우터 업데이트
        self.meta_router.update_routing_weights({
            'module_id': result.get('selected_module'),
            'performance': feedback
        })
        
        # 의사결정 엔진 업데이트
        self.decision_engine.update_decision_policy(feedback)
        
        # 인지 상태 업데이트
        if 'cognitive_feedback' in feedback:
            self.cognitive_state.update_state(feedback['cognitive_feedback'])
            
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        시스템 통계를 반환합니다.
        
        Returns:
            시스템 통계 딕셔너리
        """
        return {
            'system_status': {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'uptime': self._get_uptime()
            },
            'cognitive_state': self.cognitive_state.get_cognitive_state_summary(),
            'working_memory': self.working_memory.get_memory_statistics(),
            'meta_router': self.meta_router.get_routing_statistics(),
            'decision_engine': self.decision_engine.get_decision_statistics(),
            'attention_control': self.attention_control.get_attention_state(),
            'message_passing': self.message_passing.get_message_statistics(),
            'bio_agrpo': self.bio_agrpo.get_training_statistics(),
            'module_registry': {
                module_id: {
                    'status': info['status'],
                    'last_activity': info['last_activity']
                }
                for module_id, info in self.module_registry.items()
            }
        }
        
    def _get_uptime(self) -> Optional[float]:
        """시스템 가동 시간을 반환합니다."""
        if self.system_start_time:
            return (datetime.now() - self.system_start_time).total_seconds()
        return None
        
    def start(self):
        """시스템을 시작합니다."""
        if not self.is_initialized:
            raise RuntimeError("BioHama 시스템이 초기화되지 않았습니다.")
            
        self.is_running = True
        self.system_start_time = datetime.now()
        self.logger.info("BioHama 시스템이 시작되었습니다.")
        
    def stop(self):
        """시스템을 중지합니다."""
        self.is_running = False
        self.logger.info("BioHama 시스템이 중지되었습니다.")
        
    def reset(self):
        """시스템을 초기화합니다."""
        self.logger.info("BioHama 시스템을 초기화합니다.")
        
        # 모든 모듈 초기화
        for module in self.modules.values():
            module.reset()
            
        # 시스템 상태 초기화
        self.is_running = False
        self.system_start_time = None
        
        self.logger.info("BioHama 시스템이 초기화되었습니다.")
        
    def save_checkpoint(self, filepath: str):
        """
        시스템 체크포인트를 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        import pickle
        
        checkpoint_data = {
            'config': self.config,
            'system_state': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'system_start_time': self.system_start_time
            },
            'modules': {
                module_id: module.get_state()
                for module_id, module in self.modules.items()
            },
            'module_registry': self.module_registry
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.logger.info(f"체크포인트가 저장되었습니다: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        시스템 체크포인트를 로드합니다.
        
        Args:
            filepath: 로드할 파일 경로
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        # 설정 업데이트
        self.config.update(checkpoint_data.get('config', {}))
        
        # 시스템 상태 복원
        system_state = checkpoint_data.get('system_state', {})
        self.is_initialized = system_state.get('is_initialized', False)
        self.is_running = system_state.get('is_running', False)
        self.system_start_time = system_state.get('system_start_time')
        
        # 모듈 상태 복원
        modules_state = checkpoint_data.get('modules', {})
        for module_id, state in modules_state.items():
            if module_id in self.modules:
                self.modules[module_id].update_state(state)
                
        # 모듈 레지스트리 복원
        self.module_registry = checkpoint_data.get('module_registry', {})
        
        self.logger.info(f"체크포인트가 로드되었습니다: {filepath}")
        
    def __str__(self) -> str:
        """시스템 문자열 표현"""
        return f"BioHamaSystem(initialized={self.is_initialized}, running={self.is_running})"
        
    def __repr__(self) -> str:
        """시스템 상세 문자열 표현"""
        return f"BioHamaSystem(config={self.config}, device={self.device})"
