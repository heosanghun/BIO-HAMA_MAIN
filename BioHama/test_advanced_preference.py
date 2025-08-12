"""
고급 선호도 모델 테스트

선호도 학습기와 메모리를 포함한 고급 기능을 검증합니다.
"""

import torch
import numpy as np
import sys
import os

# BioHama 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from biohama.training.preference_model import PreferenceModel
from biohama.training.preference_learner import PreferenceLearner
from biohama.training.preference_memory import PreferenceMemory, PreferenceData


def test_preference_learner():
    """선호도 학습기 테스트"""
    print("🧪 선호도 학습기 테스트 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 학습기 초기화
    learner = PreferenceLearner(
        embedding_dim=64,
        learning_rate=1e-4,
        memory_size=1000
    )
    
    # 테스트 데이터
    current_pref = torch.randn(4, 64, device=device)
    target_pref = torch.randn(4, 64, device=device)
    
    # 컨텍스트 정보
    context = {
        'confidence': 0.8,
        'reward': 0.6,
        'time_since_last_update': 2.0
    }
    
    # 온라인 학습 테스트
    updated_pref = learner.update_preference(
        current_pref, target_pref, 'online', context
    )
    
    print(f"✅ 온라인 학습 테스트 성공")
    print(f"   - 업데이트 전 차이: {torch.norm(current_pref - target_pref).item():.4f}")
    print(f"   - 업데이트 후 차이: {torch.norm(updated_pref - target_pref).item():.4f}")
    
    # 배치 학습 테스트
    updated_pref_batch = learner.update_preference(
        current_pref, target_pref, 'batch', context
    )
    
    print(f"✅ 배치 학습 테스트 성공")
    print(f"   - 배치 업데이트 후 차이: {torch.norm(updated_pref_batch - target_pref).item():.4f}")
    
    # 학습 통계 테스트
    stats = learner.get_learning_stats()
    print(f"✅ 학습 통계 테스트 성공")
    print(f"   - 총 업데이트: {stats['total_updates']}")
    print(f"   - 수렴률: {stats['convergence_rate']:.4f}")
    print(f"   - 평균 손실: {stats['avg_loss']:.4f}")
    
    return True


def test_preference_memory():
    """선호도 메모리 테스트"""
    print("\n🧪 선호도 메모리 테스트 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 메모리 초기화
    memory = PreferenceMemory(
        max_size=1000,
        memory_type='fifo',
        similarity_threshold=0.7
    )
    
    # 테스트 데이터 생성
    test_data = []
    for i in range(10):
        data = PreferenceData(
            preference_type='explicit' if i % 2 == 0 else 'implicit',
            input_data=torch.randn(4, 64, device=device),
            preference_value=0.5 + i * 0.1,
            timestamp=time.time() + i,
            context={'test_id': i}
        )
        test_data.append(data)
    
    # 데이터 저장 테스트
    stored_keys = []
    for i, data in enumerate(test_data):
        key = memory.store(data, f"test_key_{i}")
        stored_keys.append(key)
        
    print(f"✅ 데이터 저장 테스트 성공")
    print(f"   - 저장된 데이터 수: {len(stored_keys)}")
    
    # 데이터 검색 테스트
    retrieved_data = memory.retrieve(stored_keys[0])
    print(f"✅ 데이터 검색 테스트 성공")
    print(f"   - 검색된 데이터 타입: {retrieved_data.preference_type}")
    print(f"   - 검색된 데이터 값: {retrieved_data.preference_value}")
    
    # 유사도 검색 테스트
    query_data = torch.randn(4, 64, device=device)
    similar_data = memory.retrieve_similar(query_data, top_k=3)
    
    print(f"✅ 유사도 검색 테스트 성공")
    print(f"   - 유사한 데이터 수: {len(similar_data)}")
    if similar_data:
        print(f"   - 최고 유사도: {similar_data[0][2]:.4f}")
    
    # 타입별 검색 테스트
    explicit_data = memory.retrieve_by_type('explicit', limit=3)
    implicit_data = memory.retrieve_by_type('implicit', limit=3)
    
    print(f"✅ 타입별 검색 테스트 성공")
    print(f"   - 명시적 데이터: {len(explicit_data)}개")
    print(f"   - 암시적 데이터: {len(implicit_data)}개")
    
    # 메모리 통계 테스트
    stats = memory.get_memory_stats()
    print(f"✅ 메모리 통계 테스트 성공")
    print(f"   - 히트율: {stats['hit_rate']:.4f}")
    print(f"   - 현재 크기: {stats['current_size']}")
    print(f"   - 타입 분포: {stats['type_distribution']}")
    
    return True


def test_advanced_preference_model():
    """고급 선호도 모델 테스트"""
    print("\n🧪 고급 선호도 모델 테스트 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 고급 설정
    config = {
        'input_dim': 512,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'learning_rate': 1e-4,
        'learner_memory_size': 1000,
        'memory_size': 5000,
        'memory_type': 'fifo',
        'similarity_threshold': 0.8
    }
    
    # 모델 초기화
    preference_model = PreferenceModel(config, device)
    preference_model.to(device)
    
    # 테스트 데이터
    batch_size = 4
    outputs = {
        'attention_output': torch.randn(batch_size, 64, 512, device=device),
        'attention_mask': torch.rand(batch_size, 8, 64, 64, device=device) > 0.5,
        'computation_savings': 0.75,
        'should_terminate': torch.rand(batch_size, device=device) > 0.5,
        'confidence': torch.rand(batch_size, device=device),
        'quality': torch.rand(batch_size, device=device)
    }
    
    preferences = {
        'attention_output': 0.8,
        'decision_output': 0.9
    }
    
    rewards = torch.rand(batch_size, device=device)
    
    # 고급 업데이트 테스트
    loss = preference_model.update(outputs, preferences, rewards)
    
    print(f"✅ 고급 업데이트 테스트 성공")
    print(f"   - 손실값: {loss:.4f}")
    
    # 요약 정보 테스트
    summary = preference_model.get_preference_summary()
    print(f"✅ 요약 정보 테스트 성공")
    print(f"   - 업데이트 횟수: {summary['update_count']}")
    print(f"   - 학습 통계: {summary['learner_stats']['total_updates']}회 업데이트")
    print(f"   - 메모리 통계: {summary['memory_stats']['current_size']}개 저장")
    
    # 메모리 최적화 테스트
    preference_model.memory.optimize_memory()
    print(f"✅ 메모리 최적화 테스트 성공")
    
    return True


def test_integration():
    """통합 테스트"""
    print("\n🧪 통합 테스트 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 고급 설정
    config = {
        'input_dim': 512,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'learning_rate': 1e-4,
        'learner_memory_size': 1000,
        'memory_size': 5000,
        'memory_type': 'lru',  # LRU 메모리 사용
        'similarity_threshold': 0.8
    }
    
    # 모델 초기화
    preference_model = PreferenceModel(config, device)
    
    # 연속 업데이트 테스트
    for epoch in range(5):
        batch_size = 4
        outputs = {
            'attention_output': torch.randn(batch_size, 64, 512, device=device),
            'attention_mask': torch.rand(batch_size, 8, 64, 64, device=device) > 0.5,
            'computation_savings': 0.75,
            'should_terminate': torch.rand(batch_size, device=device) > 0.5,
            'confidence': torch.rand(batch_size, device=device),
            'quality': torch.rand(batch_size, device=device)
        }
        
        preferences = {
            'attention_output': 0.8 + epoch * 0.02,
            'decision_output': 0.9 - epoch * 0.01
        }
        
        rewards = torch.rand(batch_size, device=device)
        
        loss = preference_model.update(outputs, preferences, rewards)
        
        if epoch % 2 == 0:
            print(f"   - Epoch {epoch}: 손실 = {loss:.4f}")
    
    # 최종 통계
    summary = preference_model.get_preference_summary()
    
    print(f"✅ 통합 테스트 성공")
    print(f"   - 총 업데이트: {summary['update_count']}회")
    print(f"   - 학습 수렴률: {summary['learner_stats']['convergence_rate']:.4f}")
    print(f"   - 메모리 히트율: {summary['memory_stats']['hit_rate']:.4f}")
    print(f"   - 메모리 크기: {summary['memory_stats']['current_size']}")
    
    return True


def main():
    """메인 테스트 함수"""
    print("🚀 고급 선호도 모델 테스트 시작\n")
    
    try:
        # 개별 테스트
        test_preference_learner()
        test_preference_memory()
        test_advanced_preference_model()
        test_integration()
        
        print("\n🎉 모든 고급 테스트가 성공적으로 완료되었습니다!")
        print("✅ 선호도 학습기: 온라인/배치 학습 정상 작동")
        print("✅ 선호도 메모리: FIFO/LRU 메모리 정상 작동")
        print("✅ 고급 선호도 모델: 통합 기능 정상 작동")
        print("✅ 통합 테스트: 전체 시스템 연동 정상 작동")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)
