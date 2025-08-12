#!/usr/bin/env python3
"""
BioHama 시스템 데모

이 스크립트는 BioHama 시스템의 기본 기능을 보여줍니다.
"""

import sys
import os
import time

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_demo_config():
    """데모용 설정 생성"""
    return {
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

def print_banner():
    """배너 출력"""
    print("=" * 60)
    print("🧠 BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처")
    print("=" * 60)
    print()

def demo_system_initialization():
    """시스템 초기화 데모"""
    print("📋 1단계: 시스템 초기화")
    print("-" * 40)
    
    try:
        from biohama import BioHamaSystem
        
        config = create_demo_config()
        print("✅ 설정 생성 완료")
        
        print("🔄 BioHama 시스템 초기화 중...")
        biohama = BioHamaSystem(config)
        print("✅ BioHama 시스템 초기화 완료!")
        
        return biohama
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return None

def demo_basic_processing(biohama):
    """기본 처리 데모"""
    print("\n📋 2단계: 기본 입력 처리")
    print("-" * 40)
    
    if biohama is None:
        print("❌ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        biohama.start()
        print("✅ 시스템 시작됨")
        
        # 다양한 입력 처리
        test_inputs = [
            {'text': '안녕하세요, BioHama 시스템입니다!', 'task_type': 'greeting'},
            {'text': '간단한 계산을 해주세요: 2 + 3 = ?', 'task_type': 'calculation'},
            {'text': '오늘 날씨는 어떤가요?', 'task_type': 'weather_inquiry'},
            {'text': '머신러닝에 대해 설명해주세요.', 'task_type': 'explanation'},
            {'text': '감정 분석을 해주세요.', 'task_type': 'sentiment_analysis'}
        ]
        
        for i, inputs in enumerate(test_inputs, 1):
            print(f"\n🔍 입력 {i} 처리 중...")
            print(f"   입력: {inputs['text']}")
            
            result = biohama.process_input(inputs)
            
            print(f"   ✅ 처리 완료")
            print(f"   - 선택된 모듈: {result.get('selected_module', 'N/A')}")
            print(f"   - 신뢰도: {result.get('routing_confidence', 0.0):.3f}")
            
            if 'error' in result:
                print(f"   - 오류: {result['error']}")
            
            time.sleep(0.5)  # 시각적 효과를 위한 지연
        
        biohama.stop()
        print("\n✅ 기본 처리 데모 완료!")
        
    except Exception as e:
        print(f"❌ 기본 처리 실패: {e}")

def demo_system_statistics(biohama):
    """시스템 통계 데모"""
    print("\n📋 3단계: 시스템 통계 확인")
    print("-" * 40)
    
    if biohama is None:
        print("❌ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        biohama.start()
        
        # 일부 입력 처리
        for i in range(5):
            biohama.process_input({
                'text': f'통계 테스트 입력 {i+1}',
                'task_type': 'test'
            })
        
        # 통계 수집
        stats = biohama.get_system_statistics()
        
        print("📊 시스템 통계:")
        print(f"   - 시스템 상태: {'초기화됨' if stats['system_status']['initialized'] else '초기화 안됨'}")
        print(f"   - 실행 상태: {'실행 중' if stats['system_status']['running'] else '중지됨'}")
        print(f"   - 가동 시간: {stats['system_status'].get('uptime', 0):.1f}초")
        
        print("\n🧠 인지 상태:")
        cognitive = stats['cognitive_state']
        print(f"   - 작업 메모리 사용률: {cognitive.get('working_memory_utilization', 0):.2f}")
        print(f"   - 주의 초점: {cognitive.get('attention_focus', 0):.2f}")
        print(f"   - 감정 가치: {cognitive.get('emotion_valence', 0):.2f}")
        
        print("\n💾 작업 메모리:")
        memory = stats['working_memory']
        print(f"   - 총 항목 수: {memory.get('total_items', 0)}")
        print(f"   - 사용률: {memory.get('utilization', 0):.2f}")
        print(f"   - 평균 우선순위: {memory.get('avg_priority', 0):.2f}")
        
        print("\n🔄 메타 라우터:")
        router = stats['meta_router']
        print(f"   - 총 결정 수: {router.get('total_decisions', 0)}")
        print(f"   - 탐색률: {router.get('exploration_rate', 0):.2f}")
        
        biohama.stop()
        print("\n✅ 시스템 통계 데모 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 통계 확인 실패: {e}")

def demo_training(biohama):
    """훈련 데모"""
    print("\n📋 4단계: 간단한 훈련 데모")
    print("-" * 40)
    
    if biohama is None:
        print("❌ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        biohama.start()
        
        # 간단한 훈련 데이터 생성
        training_data = []
        for i in range(10):
            training_data.append({
                'input': {
                    'text': f'훈련 샘플 {i+1}',
                    'task_type': 'training'
                },
                'context': {'training_step': i},
                'feedback': {
                    'reward': 0.7 + (i % 3) * 0.1,  # 0.7 ~ 0.9
                    'success': i % 2 == 0,  # 번갈아가며 성공/실패
                    'performance_score': 0.6 + (i % 4) * 0.1  # 0.6 ~ 0.9
                }
            })
        
        print(f"📚 {len(training_data)}개의 훈련 샘플 준비 완료")
        
        # 훈련 실행
        print("🔄 훈련 시작...")
        result = biohama.train(training_data)
        
        print("✅ 훈련 완료!")
        print(f"   - 처리된 샘플 수: {result['training_samples']}")
        print(f"   - 학습 결과 수: {len(result['learning_results'])}")
        
        biohama.stop()
        print("\n✅ 훈련 데모 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 실패: {e}")

def demo_checkpoint(biohama):
    """체크포인트 데모"""
    print("\n📋 5단계: 체크포인트 데모")
    print("-" * 40)
    
    if biohama is None:
        print("❌ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        biohama.start()
        
        # 일부 입력 처리
        biohama.process_input({'text': '체크포인트 테스트', 'task_type': 'test'})
        
        # 체크포인트 저장
        checkpoint_path = 'biohama_demo_checkpoint.pkl'
        print(f"💾 체크포인트 저장 중: {checkpoint_path}")
        biohama.save_checkpoint(checkpoint_path)
        print("✅ 체크포인트 저장 완료")
        
        # 시스템 재초기화
        biohama.stop()
        print("🔄 시스템 재초기화...")
        
        # 새 시스템 생성 및 체크포인트 로드
        from biohama import BioHamaSystem
        config = create_demo_config()
        biohama2 = BioHamaSystem(config)
        biohama2.load_checkpoint(checkpoint_path)
        print("✅ 체크포인트 로드 완료")
        
        # 동일한 입력으로 테스트
        biohama2.start()
        result = biohama2.process_input({'text': '체크포인트 테스트', 'task_type': 'test'})
        print("✅ 체크포인트 복원 테스트 완료")
        
        biohama2.stop()
        
        # 임시 파일 정리
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("🧹 임시 체크포인트 파일 정리 완료")
        
        print("\n✅ 체크포인트 데모 완료!")
        
    except Exception as e:
        print(f"❌ 체크포인트 데모 실패: {e}")

def main():
    """메인 데모 함수"""
    print_banner()
    
    print("🚀 BioHama 시스템 데모를 시작합니다...")
    print()
    
    # 1. 시스템 초기화
    biohama = demo_system_initialization()
    
    if biohama is None:
        print("\n❌ 시스템 초기화에 실패했습니다. 데모를 종료합니다.")
        return
    
    # 2. 기본 처리
    demo_basic_processing(biohama)
    
    # 3. 시스템 통계
    demo_system_statistics(biohama)
    
    # 4. 훈련
    demo_training(biohama)
    
    # 5. 체크포인트
    demo_checkpoint(biohama)
    
    print("\n" + "=" * 60)
    print("🎉 BioHama 시스템 데모 완료!")
    print("=" * 60)
    print("\n📝 데모 요약:")
    print("   ✅ 시스템 초기화 및 구성 요소 확인")
    print("   ✅ 다양한 입력 처리 및 라우팅")
    print("   ✅ 시스템 통계 및 모니터링")
    print("   ✅ 간단한 훈련 및 학습")
    print("   ✅ 체크포인트 저장/로드")
    print("\n🔬 BioHama는 뇌과학적 기반의 인공지능 시스템으로,")
    print("   생물학적 신경망의 적응성과 학습 메커니즘을 모방합니다.")

if __name__ == "__main__":
    main()
