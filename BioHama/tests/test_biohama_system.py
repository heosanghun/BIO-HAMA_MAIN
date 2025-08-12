"""
BioHama 시스템 테스트

BioHama 시스템의 주요 기능들을 테스트합니다.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from typing import Dict, Any

# 테스트를 위한 경로 설정
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from biohama import BioHamaSystem


class TestBioHamaSystem(unittest.TestCase):
    """BioHama 시스템 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.config = {
            'device': 'cpu',
            'meta_router': {
                'input_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'routing_temperature': 1.0,
                'exploration_rate': 0.1,
                'confidence_threshold': 0.7
            },
            'cognitive_state': {
                'working_memory_dim': 128,
                'attention_dim': 64,
                'emotion_dim': 32,
                'metacognitive_dim': 64,
                'max_history_length': 100
            },
            'working_memory': {
                'capacity': 128,
                'chunk_size': 32,
                'decay_rate': 0.1,
                'consolidation_threshold': 0.7
            },
            'decision_engine': {
                'decision_dim': 128,
                'num_options': 5,
                'exploration_rate': 0.1
            },
            'attention_control': {
                'attention_dim': 64,
                'num_heads': 4,
                'dropout': 0.1
            },
            'message_passing': {
                'message_dim': 64,
                'max_message_length': 100,
                'message_ttl': 5,
                'input_dim': 64,
                'output_dim': 64
            },
            'bio_agrpo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'dopamine_decay': 0.95,
                'serotonin_modulation': 0.1,
                'norepinephrine_boost': 0.2,
                'replay_buffer_size': 1000,
                'batch_size': 32,
                'state_dim': 64,
                'action_dim': 32
            }
        }
        
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        biohama = BioHamaSystem(self.config)
        
        # 기본 속성 확인
        self.assertTrue(biohama.is_initialized)
        self.assertFalse(biohama.is_running)
        self.assertEqual(biohama.device, 'cpu')
        
        # 구성 요소들이 초기화되었는지 확인
        self.assertIsNotNone(biohama.meta_router)
        self.assertIsNotNone(biohama.cognitive_state)
        self.assertIsNotNone(biohama.working_memory)
        self.assertIsNotNone(biohama.decision_engine)
        self.assertIsNotNone(biohama.attention_control)
        self.assertIsNotNone(biohama.message_passing)
        self.assertIsNotNone(biohama.bio_agrpo)
        
    def test_system_start_stop(self):
        """시스템 시작/중지 테스트"""
        biohama = BioHamaSystem(self.config)
        
        # 시작 전 상태 확인
        self.assertFalse(biohama.is_running)
        self.assertIsNone(biohama.system_start_time)
        
        # 시스템 시작
        biohama.start()
        self.assertTrue(biohama.is_running)
        self.assertIsNotNone(biohama.system_start_time)
        
        # 시스템 중지
        biohama.stop()
        self.assertFalse(biohama.is_running)
        
    def test_input_processing(self):
        """입력 처리 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 테스트 입력
        test_inputs = [
            {'text': '안녕하세요', 'task_type': 'greeting'},
            {'numbers': [1, 2, 3], 'task_type': 'calculation'},
            {'features': [0.1, 0.2, 0.3], 'task_type': 'feature_processing'}
        ]
        
        for inputs in test_inputs:
            result = biohama.process_input(inputs)
            
            # 결과 구조 확인
            self.assertIn('timestamp', result)
            self.assertIn('routing_confidence', result)
            
            # 오류가 없는지 확인
            self.assertNotIn('error', result)
            
        biohama.stop()
        
    def test_training(self):
        """훈련 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 훈련 데이터 생성
        training_data = []
        for i in range(10):
            training_data.append({
                'input': {
                    'text': f'훈련 텍스트 {i}',
                    'task_type': 'text_processing'
                },
                'context': {'training_step': i},
                'feedback': {
                    'reward': np.random.uniform(0.0, 1.0),
                    'success': np.random.choice([True, False]),
                    'performance_score': np.random.uniform(0.5, 1.0)
                }
            })
        
        # 훈련 실행
        result = biohama.train(training_data)
        
        # 훈련 결과 확인
        self.assertIn('training_samples', result)
        self.assertIn('learning_results', result)
        self.assertIn('final_statistics', result)
        self.assertEqual(result['training_samples'], 10)
        
        biohama.stop()
        
    def test_system_statistics(self):
        """시스템 통계 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 일부 입력 처리
        biohama.process_input({'text': '테스트', 'task_type': 'test'})
        
        # 통계 수집
        stats = biohama.get_system_statistics()
        
        # 통계 구조 확인
        self.assertIn('system_status', stats)
        self.assertIn('cognitive_state', stats)
        self.assertIn('working_memory', stats)
        self.assertIn('meta_router', stats)
        self.assertIn('decision_engine', stats)
        self.assertIn('attention_control', stats)
        self.assertIn('message_passing', stats)
        self.assertIn('bio_agrpo', stats)
        self.assertIn('module_registry', stats)
        
        # 시스템 상태 확인
        self.assertTrue(stats['system_status']['initialized'])
        self.assertTrue(stats['system_status']['running'])
        
        biohama.stop()
        
    def test_checkpoint_save_load(self):
        """체크포인트 저장/로드 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 일부 입력 처리
        biohama.process_input({'text': '체크포인트 테스트', 'task_type': 'test'})
        
        # 임시 파일에 체크포인트 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            # 체크포인트 저장
            biohama.save_checkpoint(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # 새 시스템 생성 및 체크포인트 로드
            biohama2 = BioHamaSystem(self.config)
            biohama2.load_checkpoint(checkpoint_path)
            
            # 시스템이 정상적으로 로드되었는지 확인
            self.assertTrue(biohama2.is_initialized)
            
        finally:
            # 임시 파일 정리
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
                
        biohama.stop()
        
    def test_system_reset(self):
        """시스템 초기화 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 일부 입력 처리
        biohama.process_input({'text': '리셋 테스트', 'task_type': 'test'})
        
        # 시스템 초기화
        biohama.reset()
        
        # 초기화 후 상태 확인
        self.assertTrue(biohama.is_initialized)
        self.assertFalse(biohama.is_running)
        self.assertIsNone(biohama.system_start_time)
        
    def test_error_handling(self):
        """오류 처리 테스트"""
        # 잘못된 설정으로 시스템 초기화 시도
        invalid_config = {
            'device': 'cpu',
            'meta_router': {
                'input_dim': -1  # 잘못된 값
            }
        }
        
        # 초기화 중 오류가 발생하는지 확인
        with self.assertRaises(Exception):
            BioHamaSystem(invalid_config)
            
    def test_memory_management(self):
        """메모리 관리 테스트"""
        biohama = BioHamaSystem(self.config)
        biohama.start()
        
        # 여러 입력 처리로 메모리 사용량 증가
        for i in range(20):
            biohama.process_input({
                'text': f'메모리 테스트 {i}',
                'task_type': 'memory_test'
            })
        
        # 메모리 통계 확인
        stats = biohama.get_system_statistics()
        memory_stats = stats['working_memory']
        
        # 메모리 사용량이 증가했는지 확인
        self.assertGreater(memory_stats['total_items'], 0)
        self.assertGreater(memory_stats['utilization'], 0.0)
        
        biohama.stop()
        
    def test_module_registry(self):
        """모듈 레지스트리 테스트"""
        biohama = BioHamaSystem(self.config)
        
        # 모듈 레지스트리 확인
        registry = biohama.module_registry
        
        # 필수 모듈들이 등록되어 있는지 확인
        required_modules = [
            'meta_router', 'cognitive_state', 'working_memory',
            'decision_engine', 'attention_control', 'message_passing', 'bio_agrpo'
        ]
        
        for module_name in required_modules:
            self.assertIn(module_name, registry)
            self.assertEqual(registry[module_name]['status'], 'active')
            self.assertIn('last_activity', registry[module_name])


class TestBioHamaComponents(unittest.TestCase):
    """BioHama 구성 요소들 개별 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.config = {
            'device': 'cpu',
            'meta_router': {
                'input_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'routing_temperature': 1.0,
                'exploration_rate': 0.1,
                'confidence_threshold': 0.7
            },
            'cognitive_state': {
                'working_memory_dim': 128,
                'attention_dim': 64,
                'emotion_dim': 32,
                'metacognitive_dim': 64,
                'max_history_length': 100
            },
            'working_memory': {
                'capacity': 128,
                'chunk_size': 32,
                'decay_rate': 0.1,
                'consolidation_threshold': 0.7
            },
            'decision_engine': {
                'decision_dim': 128,
                'num_options': 5,
                'exploration_rate': 0.1
            },
            'attention_control': {
                'attention_dim': 64,
                'num_heads': 4,
                'dropout': 0.1
            },
            'message_passing': {
                'message_dim': 64,
                'max_message_length': 100,
                'message_ttl': 5,
                'input_dim': 64,
                'output_dim': 64
            },
            'bio_agrpo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'dopamine_decay': 0.95,
                'serotonin_modulation': 0.1,
                'norepinephrine_boost': 0.2,
                'replay_buffer_size': 1000,
                'batch_size': 32,
                'state_dim': 64,
                'action_dim': 32
            }
        }
        
    def test_meta_router(self):
        """메타 라우터 테스트"""
        from biohama import MetaRouter
        
        router = MetaRouter(self.config['meta_router'])
        
        # 라우팅 테스트
        inputs = {'text': '테스트 입력', 'task_type': 'test'}
        available_modules = [router]  # 자기 자신을 모듈로 사용
        
        result = router.route(inputs, available_modules)
        
        self.assertIn('selected_module', result)
        self.assertIn('confidence', result)
        self.assertIn('exploration', result)
        
    def test_cognitive_state(self):
        """인지 상태 관리 테스트"""
        from biohama import CognitiveState
        
        cognitive_state = CognitiveState(self.config['cognitive_state'])
        
        # 상태 업데이트 테스트
        cognitive_state.update_working_memory(torch.randn(128), priority=0.8)
        cognitive_state.update_emotion_state(valence=0.3, arousal=0.7, dominance=0.5)
        
        # 상태 요약 확인
        summary = cognitive_state.get_cognitive_state_summary()
        self.assertIn('working_memory_utilization', summary)
        self.assertIn('emotion_valence', summary)
        
    def test_working_memory(self):
        """작업 메모리 테스트"""
        from biohama import WorkingMemory
        
        memory = WorkingMemory(self.config['working_memory'])
        
        # 메모리 저장/검색 테스트
        content = torch.randn(32)
        item_id = memory.store(content, priority=0.9)
        
        retrieved_content = memory.retrieve(item_id)
        self.assertIsNotNone(retrieved_content)
        self.assertTrue(torch.allclose(content, retrieved_content))
        
        # 통계 확인
        stats = memory.get_memory_statistics()
        self.assertIn('total_items', stats)
        self.assertIn('utilization', stats)


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)
