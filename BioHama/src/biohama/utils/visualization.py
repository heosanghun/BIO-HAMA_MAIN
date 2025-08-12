"""
시각화 모듈

BioHama 시스템의 상태 및 성능을 시각화합니다.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Optional
import seaborn as sns


def plot_system_state(state_data: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    시스템 상태 시각화
    
    Args:
        state_data: 상태 데이터
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('BioHama 시스템 상태', fontsize=16)
    
    # 인지 상태
    if 'cognitive_state' in state_data:
        cognitive = state_data['cognitive_state']
        axes[0, 0].bar(['작업메모리', '주의', '감정', '메타인지'], 
                      [cognitive.get('working_memory_utilization', 0),
                       cognitive.get('attention_focus', 0),
                       cognitive.get('emotion_valence', 0),
                       cognitive.get('metacognitive_confidence', 0)])
        axes[0, 0].set_title('인지 상태')
        axes[0, 0].set_ylim(-1, 1)
    
    # 신경전달물질
    if 'neurotransmitter' in state_data:
        nt = state_data['neurotransmitter']
        axes[0, 1].bar(['도파민', '세로토닌', '노르에피네프린'],
                      [nt.get('dopamine', 0), nt.get('serotonin', 0), nt.get('norepinephrine', 0)])
        axes[0, 1].set_title('신경전달물질 수준')
        axes[0, 1].set_ylim(0, 1)
    
    # 성능 지표
    if 'performance' in state_data:
        perf = state_data['performance']
        axes[1, 0].plot(perf.get('rewards', []))
        axes[1, 0].set_title('보상 추이')
        axes[1, 0].set_xlabel('스텝')
        axes[1, 0].set_ylabel('보상')
    
    # 모듈 활성화
    if 'module_activity' in state_data:
        activity = state_data['module_activity']
        modules = list(activity.keys())
        values = list(activity.values())
        axes[1, 1].barh(modules, values)
        axes[1, 1].set_title('모듈 활성화')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics(metrics: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    성능 지표 시각화
    
    Args:
        metrics: 성능 지표 데이터
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('BioHama 성능 지표', fontsize=16)
    
    # 손실 추이
    if 'losses' in metrics:
        losses = metrics['losses']
        axes[0, 0].plot(losses)
        axes[0, 0].set_title('손실 추이')
        axes[0, 0].set_xlabel('스텝')
        axes[0, 0].set_ylabel('손실')
    
    # 정확도 추이
    if 'accuracies' in metrics:
        accuracies = metrics['accuracies']
        axes[0, 1].plot(accuracies)
        axes[0, 1].set_title('정확도 추이')
        axes[0, 1].set_xlabel('스텝')
        axes[0, 1].set_ylabel('정확도')
    
    # 메모리 사용량
    if 'memory_usage' in metrics:
        memory = metrics['memory_usage']
        axes[1, 0].plot(memory)
        axes[1, 0].set_title('메모리 사용량')
        axes[1, 0].set_xlabel('스텝')
        axes[1, 0].set_ylabel('메모리 (MB)')
    
    # 처리 시간
    if 'processing_times' in metrics:
        times = metrics['processing_times']
        axes[1, 1].plot(times)
        axes[1, 1].set_title('처리 시간')
        axes[1, 1].set_xlabel('스텝')
        axes[1, 1].set_ylabel('시간 (초)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

