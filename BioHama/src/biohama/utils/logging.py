"""
로깅 모듈

BioHama 시스템의 구조화된 로깅을 제공합니다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> None:
    """
    로깅 설정
    
    Args:
        level: 로그 레벨
        log_file: 로그 파일 경로
        log_format: 로그 포맷
    """
    # 로거 생성
    logger = logging.getLogger('biohama')
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 생성
    formatter = logging.Formatter(log_format)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 다른 모듈들의 로그 레벨 설정
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str = "biohama") -> logging.Logger:
    """
    로거 반환
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 객체
    """
    return logging.getLogger(name)

