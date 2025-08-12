"""
설정 관리 모듈

BioHama 시스템의 설정을 로드하고 관리합니다.
"""

import yaml
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"지원하지 않는 설정 파일 형식: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 저장할 파일 경로
    """
    config_path = Path(config_path)
    
    # 디렉토리 생성
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"지원하지 않는 설정 파일 형식: {config_path.suffix}")


def get_config(config_name: str = "base_config") -> Dict[str, Any]:
    """
    기본 설정 반환
    
    Args:
        config_name: 설정 이름
        
    Returns:
        기본 설정 딕셔너리
    """
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    
    # 설정 파일 경로
    config_path = project_root / "configs" / f"{config_name}.yaml"
    
    if config_path.exists():
        return load_config(config_path)
    else:
        # 기본 설정 반환
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    기본 설정 반환
    
    Returns:
        기본 설정 딕셔너리
    """
    return {
        # 시스템 설정
        'system': {
            'name': 'BioHama',
            'version': '0.1.0',
            'device': 'auto',
            'seed': 42
        },
        
        # 핵심 아키텍처 설정
        'core': {
            'meta_router': {
                'input_dim': 512,
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
                'metacognitive_dim': 128
            },
            'working_memory': {
                'capacity': 256,
                'chunk_size': 64,
                'decay_rate': 0.1,
                'consolidation_threshold': 0.7
            }
        },
        
        # 학습 시스템 설정
        'learning': {
            'bio_agrpo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'replay_buffer_size': 10000,
                'batch_size': 64
            },
            'neurotransmitter': {
                'dopamine_decay': 0.95,
                'serotonin_modulation': 0.1,
                'norepinephrine_boost': 0.2
            }
        },
        
        # 통신 시스템 설정
        'communication': {
            'message_passing': {
                'message_dim': 128,
                'max_message_length': 1000,
                'message_ttl': 10
            },
            'attention_graph': {
                'max_nodes': 100,
                'attention_dim': 128,
                'decay_rate': 0.95
            },
            'hebbian_learning': {
                'learning_rate': 0.01,
                'decay_rate': 0.99,
                'connection_threshold': 0.1
            },
            'temporal_credit': {
                'credit_decay': 0.9,
                'max_temporal_window': 50,
                'eligibility_trace_decay': 0.95
            }
        },
        
        # 유틸리티 설정
        'utils': {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'biohama.log'
            },
            'profiling': {
                'enabled': True,
                'interval': 1.0
            }
        }
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 병합
    
    Args:
        base_config: 기본 설정
        override_config: 오버라이드 설정
        
    Returns:
        병합된 설정
    """
    def deep_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(base_config, override_config)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    설정 유효성 검사
    
    Args:
        config: 검사할 설정
        
    Returns:
        유효성 여부
    """
    required_keys = ['system', 'core', 'learning', 'communication', 'utils']
    
    for key in required_keys:
        if key not in config:
            print(f"필수 설정 키가 없습니다: {key}")
            return False
    
    return True

