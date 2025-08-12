"""
디바이스 유틸리티 모듈

BioHama 시스템의 GPU/CPU 관리를 제공합니다.
"""

import torch
from typing import Any, Union, Optional


def get_device(device_preference: str = "auto") -> torch.device:
    """
    사용할 디바이스 반환
    
    Args:
        device_preference: 디바이스 선호도 ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch 디바이스
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """
    데이터를 지정된 디바이스로 이동
    
    Args:
        data: 이동할 데이터
        device: 대상 디바이스
        
    Returns:
        디바이스로 이동된 데이터
    """
    if isinstance(device, str):
        device = get_device(device)
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data


def get_device_info(device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    디바이스 정보 반환
    
    Args:
        device: 확인할 디바이스
        
    Returns:
        디바이스 정보 딕셔너리
    """
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = get_device(device)
    
    info = {
        'device': str(device),
        'type': device.type
    }
    
    if device.type == 'cuda':
        info.update({
            'name': torch.cuda.get_device_name(device),
            'memory_total_gb': torch.cuda.get_device_properties(device).total_memory / 1024**3,
            'memory_allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
            'memory_cached_gb': torch.cuda.memory_reserved(device) / 1024**3,
            'compute_capability': torch.cuda.get_device_capability(device)
        })
    elif device.type == 'mps':
        info.update({
            'name': 'Apple Silicon GPU',
            'available': True
        })
    else:
        info.update({
            'name': 'CPU',
            'available': True
        })
    
    return info


def clear_gpu_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    GPU 메모리 정리
    
    Args:
        device: 정리할 디바이스
    """
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = get_device(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def set_device_seed(seed: int, device: Optional[Union[str, torch.device]] = None) -> None:
    """
    디바이스별 시드 설정
    
    Args:
        seed: 시드 값
        device: 대상 디바이스
    """
    torch.manual_seed(seed)
    
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = get_device(device)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 재현성을 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

