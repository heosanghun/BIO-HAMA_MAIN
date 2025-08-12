#!/usr/bin/env python3
"""
간단한 BioHama 시스템 테스트
"""

import sys
import os

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """기본 import 테스트"""
    print("=== BioHama Import 테스트 ===")
    
    try:
        print("1. 기본 모듈 import 시도...")
        from biohama import BioHamaSystem
        print("✅ BioHamaSystem import 성공")
        
        print("2. 개별 구성 요소 import 시도...")
        from biohama import MetaRouter, CognitiveState, WorkingMemory
        print("✅ 개별 구성 요소 import 성공")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def test_basic_config():
    """기본 설정 테스트"""
    print("\n=== 기본 설정 테스트 ===")
    
    config = {
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
    
    print("✅ 기본 설정 생성 완료")
    return config

def test_system_initialization(config):
    """시스템 초기화 테스트"""
    print("\n=== 시스템 초기화 테스트 ===")
    
    try:
        from biohama import BioHamaSystem
        
        print("1. BioHama 시스템 초기화 중...")
        biohama = BioHamaSystem(config)
        print("✅ BioHama 시스템 초기화 성공")
        
        print("2. 시스템 상태 확인...")
        print(f"   - 초기화됨: {biohama.is_initialized}")
        print(f"   - 실행 중: {biohama.is_running}")
        print(f"   - 디바이스: {biohama.device}")
        
        print("3. 구성 요소 확인...")
        components = [
            ('메타 라우터', biohama.meta_router),
            ('인지 상태', biohama.cognitive_state),
            ('작업 메모리', biohama.working_memory),
            ('의사결정 엔진', biohama.decision_engine),
            ('주의 제어', biohama.attention_control),
            ('메시지 전달', biohama.message_passing),
            ('Bio-A-GRPO', biohama.bio_agrpo)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"   ✅ {name}: 초기화됨")
            else:
                print(f"   ❌ {name}: 초기화 실패")
        
        return biohama
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_basic_operations(biohama):
    """기본 작업 테스트"""
    print("\n=== 기본 작업 테스트 ===")
    
    if biohama is None:
        print("❌ 시스템이 초기화되지 않아 테스트를 건너뜁니다.")
        return
    
    try:
        print("1. 시스템 시작...")
        biohama.start()
        print("✅ 시스템 시작 성공")
        
        print("2. 기본 입력 처리...")
        test_input = {'text': '안녕하세요, BioHama!', 'task_type': 'greeting'}
        result = biohama.process_input(test_input)
        
        print("✅ 입력 처리 성공")
        print(f"   - 선택된 모듈: {result.get('selected_module', 'N/A')}")
        print(f"   - 신뢰도: {result.get('routing_confidence', 0.0):.3f}")
        
        print("3. 시스템 통계 확인...")
        stats = biohama.get_system_statistics()
        print("✅ 통계 수집 성공")
        print(f"   - 시스템 상태: {stats['system_status']['initialized']}")
        print(f"   - 작업 메모리 사용률: {stats['working_memory']['utilization']:.2f}")
        
        print("4. 시스템 중지...")
        biohama.stop()
        print("✅ 시스템 중지 성공")
        
    except Exception as e:
        print(f"❌ 기본 작업 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("🧠 BioHama 시스템 간단 테스트 시작")
    print("=" * 50)
    
    # 1. Import 테스트
    if not test_imports():
        print("\n❌ Import 테스트 실패. 종료합니다.")
        return
    
    # 2. 설정 테스트
    config = test_basic_config()
    
    # 3. 시스템 초기화 테스트
    biohama = test_system_initialization(config)
    
    # 4. 기본 작업 테스트
    test_basic_operations(biohama)
    
    print("\n" + "=" * 50)
    print("🎉 BioHama 시스템 테스트 완료!")

if __name__ == "__main__":
    main()
