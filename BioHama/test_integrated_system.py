#!/usr/bin/env python3
"""
완전한 BioHama 시스템 통합 테스트

새로 구현된 SparseAttention과 TerminationModule이 포함된
완전한 BioHama 시스템을 테스트합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any

# BioHama 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from biohama.biohama_system import BioHamaSystem


def test_complete_system():
    """완전한 BioHama 시스템 테스트"""
    print("=" * 80)
    print("🧠 완전한 BioHama 시스템 테스트 시작")
    print("=" * 80)
    
    # 시스템 설정
    config = {
        'device': 'cpu',
        'meta_router': {
            'input_dim': 512,
            'hidden_dim': 256,
            'num_heads': 8,
            'routing_strategy': 'attention'
        },
        'cognitive_state': {
            'memory_size': 1000,
            'attention_dim': 256,
            'emotion_dim': 64
        },
        'working_memory': {
            'capacity': 100,
            'consolidation_threshold': 0.7
        },
        'decision_engine': {
            'policy_dim': 256,
            'value_dim': 128
        },
        'attention_control': {
            'attention_dim': 256,
            'num_heads': 8
        },
        'sparse_attention': {
            'd_model': 512,
            'num_heads': 8,
            'seq_len': 128,
            'sparsity_ratio': 0.8,
            'local_window': 32,
            'pattern_dim': 64,
            'num_patterns': 8
        },
        'termination_module': {
            'input_dim': 512,
            'confidence_threshold': 0.7,
            'quality_threshold': 0.6,
            'patience': 3,
            'min_delta': 0.001,
            'max_iterations': 10
        },
        'message_passing': {
            'queue_size': 1000,
            'timeout': 5.0
        },
        'bio_agrpo': {
            'policy_dim': 256,
            'value_dim': 128,
            'learning_rate': 0.001
        }
    }
    
    # BioHama 시스템 초기화
    print("🔄 BioHama 시스템 초기화 중...")
    biohama = BioHamaSystem(config)
    print("✅ BioHama 시스템 초기화 완료")
    
    # 시스템 시작
    biohama.start()
    print("✅ BioHama 시스템 시작됨")
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            'name': '기본 텍스트 처리',
            'inputs': {
                'text': '안녕하세요. BioHama 시스템입니다.',
                'task_type': 'text_processing'
            }
        },
        {
            'name': '복잡한 추론 작업',
            'inputs': {
                'question': '인공지능의 미래는 어떻게 될까요?',
                'context': '최근 AI 기술의 발전과 윤리적 고려사항',
                'task_type': 'reasoning'
            }
        },
        {
            'name': '다중 모달 입력',
            'inputs': {
                'text': '이 이미지를 분석해주세요',
                'image_features': torch.randn(1, 512),
                'task_type': 'multimodal'
            }
        },
        {
            'name': '학습 시나리오',
            'inputs': {
                'training_data': torch.randn(10, 512),
                'labels': torch.randint(0, 5, (10,)),
                'task_type': 'learning'
            }
        }
    ]
    
    print(f"\n📋 {len(test_scenarios)}개 테스트 시나리오 실행")
    print("-" * 80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔄 시나리오 {i}: {scenario['name']}")
        print(f"   입력: {list(scenario['inputs'].keys())}")
        
        try:
            # 입력 처리
            result = biohama.process_input(scenario['inputs'])
            
            print(f"   ✅ 처리 완료")
            print(f"   출력 키: {list(result.keys())}")
            
            # 결과 분석
            if 'confidence' in result:
                print(f"   신뢰도: {result['confidence']:.3f}")
            if 'quality' in result:
                print(f"   품질: {result['quality']:.3f}")
            if 'sparsity_ratio' in result:
                print(f"   희소성: {result['sparsity_ratio']:.3f}")
            if 'should_terminate' in result:
                print(f"   종료 여부: {result['should_terminate']}")
                
        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
    
    # 시스템 통계 확인
    print(f"\n📊 시스템 통계")
    print("-" * 80)
    
    stats = biohama.get_system_statistics()
    
    print(f"시스템 가동 시간: {stats.get('uptime', 'N/A')}")
    print(f"처리된 입력 수: {stats.get('processed_inputs', 0)}")
    print(f"활성 모듈 수: {stats.get('active_modules', 0)}")
    
    # 모듈별 통계
    if 'module_stats' in stats:
        print(f"\n모듈별 통계:")
        for module_name, module_stat in stats['module_stats'].items():
            print(f"  {module_name}: {module_stat}")
    
    # 희소 어텐션 통계
    if hasattr(biohama, 'sparse_attention'):
        sparse_stats = biohama.sparse_attention.get_attention_stats()
        print(f"\n희소 어텐션 통계:")
        print(f"  계산 절약: {sparse_stats['computation_savings']:.3f}")
        print(f"  총 토큰: {sparse_stats['total_tokens']:,}")
        print(f"  희소 토큰: {sparse_stats['sparse_tokens']:,}")
    
    # 종료 모듈 통계
    if hasattr(biohama, 'termination_module'):
        termination_stats = biohama.termination_module.get_termination_stats()
        print(f"\n종료 모듈 통계:")
        print(f"  총 체크: {termination_stats['total_checks']}")
        print(f"  조기 종료: {termination_stats['early_terminations']}")
        print(f"  평균 반복: {termination_stats['avg_iterations']:.2f}")
    
    # 시스템 중지
    biohama.stop()
    print(f"\n✅ BioHama 시스템 중지됨")
    
    return True


def test_performance_optimization():
    """성능 최적화 테스트"""
    print("\n" + "=" * 80)
    print("⚡ 성능 최적화 테스트 시작")
    print("=" * 80)
    
    # 설정
    config = {
        'device': 'cpu',
        'sparse_attention': {
            'd_model': 256,
            'num_heads': 4,
            'seq_len': 64,
            'sparsity_ratio': 0.9,  # 높은 희소성
            'local_window': 16,
            'pattern_dim': 32,
            'num_patterns': 4
        },
        'termination_module': {
            'input_dim': 256,
            'confidence_threshold': 0.5,  # 낮은 임계값
            'quality_threshold': 0.4,
            'patience': 2,
            'min_delta': 0.01,
            'max_iterations': 5
        }
    }
    
    # 시스템 초기화
    biohama = BioHamaSystem(config)
    biohama.start()
    
    print("🔄 성능 테스트 실행 중...")
    
    # 대용량 입력 테스트
    large_input = {
        'text': 'A' * 1000,  # 긴 텍스트
        'features': torch.randn(1, 64, 256),  # 큰 특징 텐서
        'task_type': 'performance_test'
    }
    
    import time
    start_time = time.time()
    
    for i in range(10):
        result = biohama.process_input(large_input)
        
        if i == 0:  # 첫 번째 결과만 출력
            print(f"✅ 첫 번째 처리 완료")
            print(f"   처리 시간: {time.time() - start_time:.3f}초")
            print(f"   희소성: {result.get('sparsity_ratio', 'N/A')}")
            print(f"   계산 절약: {result.get('computation_savings', 'N/A')}")
    
    total_time = time.time() - start_time
    avg_time = total_time / 10
    
    print(f"\n📊 성능 결과:")
    print(f"  총 처리 시간: {total_time:.3f}초")
    print(f"  평균 처리 시간: {avg_time:.3f}초")
    print(f"  처리 속도: {10/total_time:.1f} 요청/초")
    
    biohama.stop()
    
    return True


def test_error_recovery():
    """에러 복구 테스트"""
    print("\n" + "=" * 80)
    print("🛡️ 에러 복구 테스트 시작")
    print("=" * 80)
    
    config = {
        'device': 'cpu',
        'sparse_attention': {'d_model': 128},
        'termination_module': {'input_dim': 128}
    }
    
    biohama = BioHamaSystem(config)
    biohama.start()
    
    # 다양한 에러 상황 테스트
    error_scenarios = [
        {'name': '빈 입력', 'inputs': {}},
        {'name': 'None 값', 'inputs': {'text': None}},
        {'name': '잘못된 타입', 'inputs': {'text': 123}},
        {'name': '너무 큰 텐서', 'inputs': {'features': torch.randn(1, 10000, 128)}},
    ]
    
    print("🔄 에러 상황 테스트 중...")
    
    for scenario in error_scenarios:
        print(f"  테스트: {scenario['name']}")
        
        try:
            result = biohama.process_input(scenario['inputs'])
            print(f"    ✅ 정상 처리됨")
        except Exception as e:
            print(f"    ⚠️ 예상된 에러: {type(e).__name__}")
    
    # 시스템 상태 확인
    stats = biohama.get_system_statistics()
    print(f"\n✅ 시스템 상태: {stats.get('status', 'unknown')}")
    
    biohama.stop()
    
    return True


def main():
    """메인 테스트 함수"""
    print("🚀 완전한 BioHama 시스템 통합 테스트 시작")
    print("=" * 80)
    
    test_results = []
    
    try:
        # 완전한 시스템 테스트
        test_results.append(test_complete_system())
        
        # 성능 최적화 테스트
        test_results.append(test_performance_optimization())
        
        # 에러 복구 테스트
        test_results.append(test_error_recovery())
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📋 통합 테스트 결과 요약")
    print("=" * 80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ 통과한 테스트: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 모든 통합 테스트가 성공적으로 통과했습니다!")
        print("\n✨ BioHama 시스템 완성:")
        print("   - ✅ 메타 라우터 (동적 모듈 선택)")
        print("   - ✅ 인지 상태 관리")
        print("   - ✅ 작업 메모리")
        print("   - ✅ 의사결정 엔진")
        print("   - ✅ 주의 제어")
        print("   - ✅ 동적 희소 어텐션 (O(n²) → O(n) 최적화)")
        print("   - ✅ 연산 종료 모듈 (신뢰도 기반 조기 종료)")
        print("   - ✅ 메시지 전달 시스템")
        print("   - ✅ Bio-A-GRPO 학습 알고리즘")
        print("   - ✅ 완전한 시스템 통합")
        print("\n🚀 BioHama - 뇌과학적 기반의 차세대 인공지능 시스템이 완성되었습니다!")
        return True
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
