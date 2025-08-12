"""
BioHama 버전 정보 관리
"""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

def get_version():
    """현재 버전을 반환합니다."""
    return __version__

def get_version_info():
    """버전 정보 튜플을 반환합니다."""
    return __version_info__

def is_development():
    """개발 버전인지 확인합니다."""
    return __version_info__[0] == 0
