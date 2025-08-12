"""
선호도 모델과 보상 계산기 테스트

기본 선호도 모델과 보상 계산기의 기능을 검증합니다.
"""

import torch
import numpy as np
import sys
import os

# BioHama 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from biohama.training.preference_model import PreferenceModel
from biohama.training.reward_calculator import RewardCalculator


def test_preference_model():
    """선호도 모델 테스트"""
    print("🧪 선호도 모델 테스트 시작...")
    
    # 설정
    config = {
        'input_dim': 512,
        'hidden_dim': 128,
        'embedding_dim': 64
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 초기화
    preference_model = PreferenceModel(config, device)
    preference_model.to(device)
    
    # 테스트 데이터 생성
    batch_size = 4
    input_data = torch.randn(batch_size, config['input_dim'], device=device)
    
    # 순전파 테스트
    prediction, confidence = preference_model(input_data)
    
    print(f"✅ 순전파 테스트 성공")
    print(f"   - 예측값 형태: {prediction.shape}")
    print(f"   - 신뢰도 형태: {confidence.shape}")
    print(f"   - 예측값 범위: {prediction.min().item():.4f} ~ {prediction.max().item():.4f}")
    print(f"   - 신뢰도 범위: {confidence.min().item():.4f} ~ {confidence.max().item():.4f}")
    
    # 업데이트 테스트
    outputs = {
        'attention_output': torch.randn(batch_size, 64, 512, device=device),
        'decision_output': torch.randn(batch_size, 10, device=device)
    }
    
    preferences = {
        'attention_output': 0.8,
        'decision_output': 0.9
    }
    
    loss = preference_model.update(outputs, preferences)
    
    print(f"✅ 업데이트 테스트 성공")
    print(f"   - 손실값: {loss:.4f}")
    
    # 요약 정보 테스트
    summary = preference_model.get_preference_summary()
    print(f"✅ 요약 정보 테스트 성공")
    print(f"   - 업데이트 횟수: {summary['update_count']}")
    print(f"   - 훈련 모드: {summary['training_mode']}")
    
    return True


def test_reward_calculator():
    """보상 계산기 테스트"""
    print("\n🧪 보상 계산기 테스트 시작...")
    
    # 설정
    config = {
        'accuracy_reward': {
            'accuracy_threshold': 0.8,
            'reward_scale': 1.0
        },
        'efficiency_reward': {
            'efficiency_threshold': 0.7,
            'reward_scale': 0.5
        },
        'consistency_reward': {
            'consistency_threshold': 0.6,
            'reward_scale': 0.3
        },
        'reward_weights': {
            'accuracy': 0.5,
            'efficiency': 0.3,
            'consistency': 0.2
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 계산기 초기화
    reward_calculator = RewardCalculator(config, device)
    
    # 테스트 데이터 생성
    batch_size = 4
    outputs = {
        'attention_output': torch.randn(batch_size, 64, 512, device=device),
        'attention_mask': torch.rand(batch_size, 8, 64, 64, device=device) > 0.5,
        'computation_savings': 0.75,
        'should_terminate': torch.rand(batch_size, device=device) > 0.5,
        'confidence': torch.rand(batch_size, device=device),
        'quality': torch.rand(batch_size, device=device)
    }
    
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # 보상 계산 테스트
    rewards = reward_calculator.calculate_rewards(outputs, targets)
    
    print(f"✅ 보상 계산 테스트 성공")
    print(f"   - 보상 텐서 형태: {rewards.shape}")
    print(f"   - 보상 범위: {rewards.min().item():.4f} ~ {rewards.max().item():.4f}")
    print(f"   - 평균 보상: {rewards.mean().item():.4f}")
    
    # 선호도 기반 보상 테스트
    preferences = {
        'accuracy': 0.8,
        'efficiency': 0.6,
        'consistency': 0.7
    }
    
    rewards_with_prefs = reward_calculator.calculate_rewards(outputs, targets, preferences)
    
    print(f"✅ 선호도 기반 보상 테스트 성공")
    print(f"   - 선호도 적용 전 평균: {rewards.mean().item():.4f}")
    print(f"   - 선호도 적용 후 평균: {rewards_with_prefs.mean().item():.4f}")
    
    # 요약 정보 테스트
    summary = reward_calculator.get_reward_summary()
    print(f"✅ 요약 정보 테스트 성공")
    print(f"   - 총 계산 횟수: {summary['reward_stats']['total_calculations']}")
    print(f"   - 평균 정확도 보상: {summary['reward_stats']['avg_accuracy_reward']:.4f}")
    print(f"   - 평균 효율성 보상: {summary['reward_stats']['avg_efficiency_reward']:.4f}")
    print(f"   - 평균 일관성 보상: {summary['reward_stats']['avg_consistency_reward']:.4f}")
    
    return True


def test_integration():
    """통합 테스트"""
    print("\n🧪 통합 테스트 시작...")
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    preference_config = {
        'input_dim': 512,
        'hidden_dim': 128,
        'embedding_dim': 64
    }
    
    reward_config = {
        'accuracy_reward': {'accuracy_threshold': 0.8, 'reward_scale': 1.0},
        'efficiency_reward': {'efficiency_threshold': 0.7, 'reward_scale': 0.5},
        'consistency_reward': {'consistency_threshold': 0.6, 'reward_scale': 0.3},
        'reward_weights': {'accuracy': 0.5, 'efficiency': 0.3, 'consistency': 0.2}
    }
    
    # 모델 초기화
    preference_model = PreferenceModel(preference_config, device)
    reward_calculator = RewardCalculator(reward_config, device)
    
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
    
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # 1. 보상 계산
    rewards = reward_calculator.calculate_rewards(outputs, targets)
    
    # 2. 선호도 모델 업데이트
    preferences = {
        'attention_output': 0.8,
        'decision_output': 0.9
    }
    
    preference_loss = preference_model.update(outputs, preferences, rewards)
    
    print(f"✅ 통합 테스트 성공")
    print(f"   - 계산된 보상: {rewards.mean().item():.4f}")
    print(f"   - 선호도 손실: {preference_loss:.4f}")
    
    # 3. 모델 요약 정보
    pref_summary = preference_model.get_preference_summary()
    reward_summary = reward_calculator.get_reward_summary()
    
    print(f"   - 선호도 모델 업데이트: {pref_summary['update_count']}회")
    print(f"   - 보상 계산 횟수: {reward_summary['reward_stats']['total_calculations']}회")
    
    return True


def main():
    """메인 테스트 함수"""
    print("🚀 선호도 모델과 보상 계산기 테스트 시작\n")
    
    try:
        # 개별 테스트
        test_preference_model()
        test_reward_calculator()
        test_integration()
        
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("✅ 선호도 모델: 기본 기능 정상 작동")
        print("✅ 보상 계산기: 기본 기능 정상 작동")
        print("✅ 통합 테스트: 모델 간 연동 정상 작동")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
