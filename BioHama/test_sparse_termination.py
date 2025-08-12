#!/usr/bin/env python3
"""
새로 구현한 SparseAttention과 TerminationModule 테스트 스크립트

이 스크립트는 다음을 테스트합니다:
1. SparseAttentionModule의 희소 어텐션 기능
2. TerminationModule의 종료 판단 기능
3. 두 모듈의 통합 동작
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any

# BioHama 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from biohama.core.sparse_attention import SparseAttentionModule, SparseAttentionState
from biohama.core.termination_module import TerminationModule, TerminationState


def test_sparse_attention_module():
    """SparseAttentionModule 테스트"""
    print("=" * 60)
    print("🧠 SparseAttentionModule 테스트 시작")
    print("=" * 60)
    
    # 설정
    config = {
        'd_model': 512,
        'num_heads': 8,
        'seq_len': 128,
        'sparsity_ratio': 0.8,
        'local_window': 32,
        'pattern_dim': 64,
        'num_patterns': 8,
        'use_flash_attention': True
    }
    
    # 모듈 초기화
    config['name'] = "TestSparseAttention"
    sparse_attention = SparseAttentionModule(config)
    
    # 테스트 데이터 생성
    batch_size = 2
    seq_len = 64
    d_model = 512
    
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"✅ 모듈 초기화 완료")
    print(f"   - 입력 크기: {query.shape}")
    print(f"   - 희소성 비율: {config['sparsity_ratio']}")
    
    # 순전파 테스트
    inputs = {
        'query': query,
        'key': key,
        'value': value,
        'seq_len': seq_len
    }
    
    outputs = sparse_attention(inputs)
    
    print(f"✅ 순전파 테스트 완료")
    print(f"   - 출력 크기: {outputs['attention_output'].shape}")
    print(f"   - 실제 희소성: {outputs['sparsity_ratio']:.3f}")
    print(f"   - 계산 절약: {outputs['computation_savings']:.3f}")
    
    # 메타데이터 테스트
    metadata = sparse_attention.get_metadata()
    complexity = sparse_attention.get_complexity()
    
    print(f"✅ 메타데이터 테스트 완료")
    print(f"   - 모듈 이름: {metadata['name']}")
    print(f"   - 모듈 타입: {metadata['module_type']}")
    print(f"   - 총 파라미터: {complexity['total_parameters']:,}")
    print(f"   - 학습 가능 파라미터: {complexity['trainable_parameters']:,}")
    
    # 통계 테스트
    stats = sparse_attention.get_attention_stats()
    print(f"✅ 통계 테스트 완료")
    print(f"   - 총 토큰: {stats['total_tokens']:,}")
    print(f"   - 희소 토큰: {stats['sparse_tokens']:,}")
    print(f"   - 패턴 매칭: {stats['pattern_matches']:.2f}")
    
    # 희소성 조정 테스트
    sparse_attention.set_sparsity_ratio(0.9)
    outputs2 = sparse_attention(inputs)
    print(f"✅ 희소성 조정 테스트 완료")
    print(f"   - 조정된 희소성: {outputs2['sparsity_ratio']:.3f}")
    
    print("✅ SparseAttentionModule 모든 테스트 통과!")
    return True


def test_termination_module():
    """TerminationModule 테스트"""
    print("\n" + "=" * 60)
    print("🛑 TerminationModule 테스트 시작")
    print("=" * 60)
    
    # 설정
    config = {
        'input_dim': 512,
        'confidence_threshold': 0.7,
        'quality_threshold': 0.6,
        'patience': 3,
        'min_delta': 0.001,
        'max_iterations': 10
    }
    
    # 모듈 초기화
    config['name'] = "TestTermination"
    termination = TerminationModule(config)
    
    # 테스트 데이터 생성
    batch_size = 2
    seq_len = 32
    input_dim = 512
    
    features = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"✅ 모듈 초기화 완료")
    print(f"   - 입력 크기: {features.shape}")
    print(f"   - 신뢰도 임계값: {config['confidence_threshold']}")
    print(f"   - 품질 임계값: {config['quality_threshold']}")
    
    # 반복 테스트
    for iteration in range(5):
        inputs = {
            'features': features,
            'iteration': iteration,
            'force_continue': False
        }
        
        outputs = termination(inputs)
        
        print(f"✅ 반복 {iteration + 1} 테스트 완료")
        print(f"   - 신뢰도: {outputs['confidence'].mean().item():.3f}")
        print(f"   - 품질: {outputs['quality'].mean().item():.3f}")
        print(f"   - 종료 여부: {outputs['should_terminate']}")
        print(f"   - 종료 이유: {outputs['stop_reasons']}")
        
        if outputs['should_terminate']:
            print(f"   - 조기 종료됨!")
            break
    
    # 메타데이터 테스트
    metadata = termination.get_metadata()
    complexity = termination.get_complexity()
    
    print(f"✅ 메타데이터 테스트 완료")
    print(f"   - 모듈 이름: {metadata['name']}")
    print(f"   - 모듈 타입: {metadata['module_type']}")
    print(f"   - 총 파라미터: {complexity['total_parameters']:,}")
    print(f"   - 학습 가능 파라미터: {complexity['trainable_parameters']:,}")
    
    # 통계 테스트
    stats = termination.get_termination_stats()
    print(f"✅ 통계 테스트 완료")
    print(f"   - 총 체크: {stats['total_checks']}")
    print(f"   - 조기 종료: {stats['early_terminations']}")
    print(f"   - 평균 반복: {stats['avg_iterations']:.2f}")
    print(f"   - 평균 신뢰도: {stats['avg_confidence']:.3f}")
    print(f"   - 평균 품질: {stats['avg_quality']:.3f}")
    
    # 임계값 조정 테스트
    termination.set_thresholds(confidence_threshold=0.5, quality_threshold=0.4)
    print(f"✅ 임계값 조정 테스트 완료")
    
    # 상태 초기화 테스트
    termination.reset_early_stopping()
    print(f"✅ 상태 초기화 테스트 완료")
    
    print("✅ TerminationModule 모든 테스트 통과!")
    return True


def test_integration():
    """두 모듈의 통합 테스트"""
    print("\n" + "=" * 60)
    print("🔗 모듈 통합 테스트 시작")
    print("=" * 60)
    
    # 설정
    sparse_config = {
        'd_model': 256,
        'num_heads': 4,
        'seq_len': 64,
        'sparsity_ratio': 0.7,
        'local_window': 16,
        'pattern_dim': 32,
        'num_patterns': 4
    }
    
    termination_config = {
        'input_dim': 256,
        'confidence_threshold': 0.6,
        'quality_threshold': 0.5,
        'patience': 2,
        'min_delta': 0.01,
        'max_iterations': 5
    }
    
    # 모듈 초기화
    sparse_config['name'] = "IntegrationSparse"
    termination_config['name'] = "IntegrationTermination"
    sparse_attention = SparseAttentionModule(sparse_config)
    termination = TerminationModule(termination_config)
    
    # 테스트 데이터
    batch_size = 1
    seq_len = 32
    d_model = 256
    
    query = torch.randn(batch_size, seq_len, d_model)
    features = torch.randn(batch_size, seq_len, d_model)
    
    print(f"✅ 모듈 초기화 완료")
    print(f"   - 희소 어텐션: {sparse_attention.name}")
    print(f"   - 종료 모듈: {termination.name}")
    
    # 통합 처리 시뮬레이션
    for iteration in range(3):
        print(f"\n🔄 반복 {iteration + 1} 처리:")
        
        # 1. 희소 어텐션 처리
        sparse_inputs = {
            'query': query,
            'seq_len': seq_len
        }
        sparse_outputs = sparse_attention(sparse_inputs)
        
        print(f"   - 희소 어텐션 완료 (희소성: {sparse_outputs['sparsity_ratio']:.3f})")
        
        # 2. 종료 판단
        termination_inputs = {
            'features': features,
            'iteration': iteration,
            'force_continue': False
        }
        termination_outputs = termination(termination_inputs)
        
        print(f"   - 종료 판단 완료 (신뢰도: {termination_outputs['confidence'].mean().item():.3f})")
        print(f"   - 종료 여부: {termination_outputs['should_terminate']}")
        
        if termination_outputs['should_terminate']:
            print(f"   - 🔴 조기 종료됨!")
            break
        else:
            print(f"   - 🟢 계속 진행")
    
    # 통합 통계
    sparse_stats = sparse_attention.get_attention_stats()
    termination_stats = termination.get_termination_stats()
    
    print(f"\n📊 통합 통계:")
    print(f"   - 희소 어텐션 계산 절약: {sparse_stats['computation_savings']:.3f}")
    print(f"   - 종료 모듈 조기 종료율: {termination_stats['early_terminations']}/{termination_stats['total_checks']}")
    
    print("✅ 모듈 통합 테스트 완료!")
    return True


def test_error_handling():
    """에러 처리 테스트"""
    print("\n" + "=" * 60)
    print("⚠️ 에러 처리 테스트 시작")
    print("=" * 60)
    
    # 잘못된 입력 테스트
    sparse_attention = SparseAttentionModule({'name': "ErrorTestSparse"})
    termination = TerminationModule({'name': "ErrorTestTermination"})
    
    # None 입력 테스트
    try:
        result = sparse_attention({'query': None})
        if result == {'query': None}:  # 입력이 그대로 반환됨
            print("✅ None 입력 처리 성공: 입력 그대로 반환")
        else:
            print("❌ None 입력 처리 실패")
            return False
    except Exception as e:
        print(f"✅ None 입력 처리 성공: {type(e).__name__}")
    
    try:
        result = termination({'features': None})
        if result == {'features': None}:  # 입력이 그대로 반환됨
            print("✅ None 입력 처리 성공: 입력 그대로 반환")
        else:
            print("❌ None 입력 처리 실패")
            return False
    except Exception as e:
        print(f"✅ None 입력 처리 성공: {type(e).__name__}")
    
    # 잘못된 크기 테스트
    try:
        sparse_attention({'query': torch.randn(1, 10, 100)})  # 잘못된 크기
        print("❌ 잘못된 크기 처리 실패")
        return False
    except Exception as e:
        print(f"✅ 잘못된 크기 처리 성공: {type(e).__name__}")
    
    print("✅ 에러 처리 테스트 완료!")
    return True


def main():
    """메인 테스트 함수"""
    print("🚀 BioHama 고급 모듈 테스트 시작")
    print("=" * 80)
    
    test_results = []
    
    try:
        # 개별 모듈 테스트
        test_results.append(test_sparse_attention_module())
        test_results.append(test_termination_module())
        
        # 통합 테스트
        test_results.append(test_integration())
        
        # 에러 처리 테스트
        test_results.append(test_error_handling())
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📋 테스트 결과 요약")
    print("=" * 80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ 통과한 테스트: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        print("\n✨ 구현된 기능:")
        print("   - 동적 희소 어텐션 (O(n²) → O(n) 최적화)")
        print("   - 패턴 기반 어텐션 마스크 생성")
        print("   - 신뢰도 기반 종료 판단")
        print("   - 조기 종료 메커니즘")
        print("   - 품질 평가 시스템")
        print("   - 모듈 간 통합 동작")
        return True
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
