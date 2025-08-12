"""
BioHama 기본 사용 예제

BioHama 시스템의 기본적인 사용법을 보여주는 예제입니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from typing import Dict, Any, List

from biohama import BioHamaSystem


def create_basic_config() -> Dict[str, Any]:
    """기본 설정을 생성합니다."""
    return {
        'device': 'cpu',
        'meta_router': {
            'input_dim': 128,
            'hidden_dim': 256,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.1,
            'routing_temperature': 1.0,
            'exploration_rate': 0.1,
            'confidence_threshold': 0.7
        },
        'cognitive_state': {
            'working_memory_dim': 256,
            'attention_dim': 128,
            'emotion_dim': 64,
            'metacognitive_dim': 128,
            'max_history_length': 1000
        },
        'working_memory': {
            'capacity': 256,
            'chunk_size': 64,
            'decay_rate': 0.1,
            'consolidation_threshold': 0.7
        },
        'decision_engine': {
            'decision_dim': 256,
            'num_options': 10,
            'exploration_rate': 0.1
        },
        'attention_control': {
            'attention_dim': 128,
            'num_heads': 8,
            'dropout': 0.1
        },
        'message_passing': {
            'message_dim': 128,
            'max_message_length': 1000,
            'message_ttl': 10,
            'input_dim': 128,
            'output_dim': 128
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
            'replay_buffer_size': 10000,
            'batch_size': 64,
            'state_dim': 128,
            'action_dim': 64
        }
    }


def example_basic_usage():
    """기본 사용 예제"""
    print("=== BioHama 기본 사용 예제 ===")
    
    # 1. 설정 생성
    config = create_basic_config()
    print("1. 설정 생성 완료")
    
    # 2. BioHama 시스템 초기화
    biohama = BioHamaSystem(config)
    print("2. BioHama 시스템 초기화 완료")
    
    # 3. 시스템 시작
    biohama.start()
    print("3. 시스템 시작 완료")
    
    # 4. 기본 입력 처리
    test_inputs = [
        {
            'text': '안녕하세요, BioHama 시스템입니다.',
            'task_type': 'greeting'
        },
        {
            'text': '간단한 계산을 해주세요: 2 + 3 = ?',
            'task_type': 'calculation'
        },
        {
            'text': '오늘 날씨는 어떤가요?',
            'task_type': 'weather_inquiry'
        }
    ]
    
    print("\n4. 입력 처리 테스트:")
    for i, inputs in enumerate(test_inputs, 1):
        print(f"\n입력 {i}: {inputs}")
        
        try:
            result = biohama.process_input(inputs)
            print(f"결과: {result.get('selected_module', 'N/A')}")
            print(f"신뢰도: {result.get('routing_confidence', 0.0):.3f}")
            
            if 'error' in result:
                print(f"오류: {result['error']}")
                
        except Exception as e:
            print(f"처리 중 오류 발생: {e}")
    
    # 5. 시스템 통계 확인
    print("\n5. 시스템 통계:")
    stats = biohama.get_system_statistics()
    
    print(f"시스템 상태: {stats['system_status']}")
    print(f"인지 상태: {stats['cognitive_state']}")
    print(f"작업 메모리: {stats['working_memory']}")
    print(f"메타 라우터: {stats['meta_router']}")
    
    # 6. 시스템 중지
    biohama.stop()
    print("\n6. 시스템 중지 완료")
    
    print("\n=== 기본 사용 예제 완료 ===")


def example_training():
    """훈련 예제"""
    print("\n=== BioHama 훈련 예제 ===")
    
    # 1. 설정 및 시스템 초기화
    config = create_basic_config()
    biohama = BioHamaSystem(config)
    biohama.start()
    
    # 2. 훈련 데이터 생성
    training_data = []
    
    for i in range(50):
        # 다양한 입력 타입 생성
        if i % 3 == 0:
            input_data = {
                'text': f'훈련 텍스트 {i}',
                'task_type': 'text_processing'
            }
        elif i % 3 == 1:
            input_data = {
                'numbers': [i, i+1, i+2],
                'task_type': 'numerical_processing'
            }
        else:
            input_data = {
                'features': torch.randn(10).tolist(),
                'task_type': 'feature_processing'
            }
        
        # 피드백 생성 (간단한 랜덤 피드백)
        feedback = {
            'reward': np.random.uniform(0.0, 1.0),
            'success': np.random.choice([True, False]),
            'performance_score': np.random.uniform(0.5, 1.0)
        }
        
        training_data.append({
            'input': input_data,
            'context': {'training_step': i},
            'feedback': feedback
        })
    
    print(f"1. 훈련 데이터 생성 완료: {len(training_data)} 개 샘플")
    
    # 3. 훈련 실행
    print("2. 훈련 시작...")
    try:
        training_result = biohama.train(training_data)
        print(f"훈련 완료: {training_result['training_samples']} 개 샘플 처리")
        print(f"학습 결과 수: {len(training_result['learning_results'])}")
        
        # 훈련 후 통계
        final_stats = training_result['final_statistics']
        print(f"최종 시스템 통계: {final_stats['system_status']}")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
    
    # 4. 시스템 중지
    biohama.stop()
    print("3. 훈련 예제 완료")


def example_checkpoint():
    """체크포인트 예제"""
    print("\n=== BioHama 체크포인트 예제 ===")
    
    # 1. 시스템 초기화
    config = create_basic_config()
    biohama = BioHamaSystem(config)
    biohama.start()
    
    # 2. 일부 입력 처리
    test_input = {'text': '체크포인트 테스트', 'task_type': 'test'}
    result = biohama.process_input(test_input)
    print(f"1. 초기 처리 결과: {result.get('selected_module', 'N/A')}")
    
    # 3. 체크포인트 저장
    checkpoint_path = 'biohama_checkpoint.pkl'
    biohama.save_checkpoint(checkpoint_path)
    print(f"2. 체크포인트 저장 완료: {checkpoint_path}")
    
    # 4. 시스템 재초기화
    biohama.stop()
    biohama2 = BioHamaSystem(config)
    biohama2.start()
    
    # 5. 체크포인트 로드
    biohama2.load_checkpoint(checkpoint_path)
    print("3. 체크포인트 로드 완료")
    
    # 6. 동일한 입력으로 테스트
    result2 = biohama2.process_input(test_input)
    print(f"4. 복원 후 처리 결과: {result2.get('selected_module', 'N/A')}")
    
    # 7. 정리
    biohama2.stop()
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("5. 임시 체크포인트 파일 삭제 완료")
    
    print("6. 체크포인트 예제 완료")


def main():
    """메인 함수"""
    print("BioHama 시스템 기본 사용 예제를 시작합니다.\n")
    
    try:
        # 기본 사용 예제
        example_basic_usage()
        
        # 훈련 예제
        example_training()
        
        # 체크포인트 예제
        example_checkpoint()
        
        print("\n모든 예제가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"예제 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

