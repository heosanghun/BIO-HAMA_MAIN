"""
BioHama 명령줄 인터페이스

BioHama 시스템을 명령줄에서 사용할 수 있는 CLI 도구입니다.
"""

import click
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .biohama_system import BioHamaSystem
from .utils.config import get_config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처"""
    pass


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='설정 파일 경로')
@click.option('--device', '-d', default='auto', help='사용할 디바이스 (cpu, cuda, auto)')
@click.option('--output', '-o', type=click.Path(), help='결과 출력 파일 경로')
def init(config: Optional[str], device: str, output: Optional[str]):
    """BioHama 시스템을 초기화하고 기본 설정을 생성합니다."""
    
    click.echo("BioHama 시스템 초기화 중...")
    
    # 디바이스 설정
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 설정 로드
    if config:
        with open(config, 'r', encoding='utf-8') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                system_config = yaml.safe_load(f)
            else:
                system_config = json.load(f)
    else:
        system_config = get_config()
    
    system_config['device'] = device
    
    # 시스템 초기화
    try:
        biohama = BioHamaSystem(system_config)
        click.echo(f"✅ BioHama 시스템이 성공적으로 초기화되었습니다. (디바이스: {device})")
        
        # 시스템 통계 출력
        stats = biohama.get_system_statistics()
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            click.echo(f"📊 시스템 통계가 {output}에 저장되었습니다.")
        else:
            click.echo("📊 시스템 통계:")
            click.echo(json.dumps(stats, indent=2, ensure_ascii=False))
            
    except Exception as e:
        click.echo(f"❌ 시스템 초기화 실패: {e}")
        raise click.Abort()


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='설정 파일 경로')
@click.option('--input', '-i', type=click.Path(exists=True), help='입력 파일 경로')
@click.option('--text', '-t', help='텍스트 입력')
@click.option('--output', '-o', type=click.Path(), help='결과 출력 파일 경로')
def process(config: str, input: Optional[str], text: Optional[str], output: Optional[str]):
    """입력을 처리하고 결과를 반환합니다."""
    
    click.echo("BioHama 시스템에서 입력 처리 중...")
    
    # 설정 로드
    with open(config, 'r', encoding='utf-8') as f:
        if config.endswith('.yaml') or config.endswith('.yml'):
            system_config = yaml.safe_load(f)
        else:
            system_config = json.load(f)
    
    # 시스템 초기화
    biohama = BioHamaSystem(system_config)
    biohama.start()
    
    try:
        # 입력 데이터 준비
        if input:
            with open(input, 'r', encoding='utf-8') as f:
                if input.endswith('.json'):
                    inputs = json.load(f)
                else:
                    inputs = {'text': f.read().strip()}
        elif text:
            inputs = {'text': text}
        else:
            click.echo("❌ 입력이 제공되지 않았습니다. --input 또는 --text 옵션을 사용하세요.")
            raise click.Abort()
        
        # 입력 처리
        result = biohama.process_input(inputs)
        
        # 결과 출력
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ 결과가 {output}에 저장되었습니다.")
        else:
            click.echo("✅ 처리 결과:")
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        click.echo(f"❌ 입력 처리 실패: {e}")
        raise click.Abort()
    finally:
        biohama.stop()


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='설정 파일 경로')
@click.option('--data', '-d', type=click.Path(exists=True), required=True, help='훈련 데이터 파일 경로')
@click.option('--epochs', '-e', default=10, help='훈련 에포크 수')
@click.option('--output', '-o', type=click.Path(), help='훈련 결과 출력 파일 경로')
def train(config: str, data: str, epochs: int, output: Optional[str]):
    """BioHama 시스템을 훈련합니다."""
    
    click.echo(f"BioHama 시스템 훈련 시작 (에포크: {epochs})...")
    
    # 설정 로드
    with open(config, 'r', encoding='utf-8') as f:
        if config.endswith('.yaml') or config.endswith('.yml'):
            system_config = yaml.safe_load(f)
        else:
            system_config = json.load(f)
    
    # 훈련 데이터 로드
    with open(data, 'r', encoding='utf-8') as f:
        if data.endswith('.json'):
            training_data = json.load(f)
        else:
            click.echo("❌ 지원되지 않는 데이터 형식입니다. JSON 파일을 사용하세요.")
            raise click.Abort()
    
    # 시스템 초기화
    biohama = BioHamaSystem(system_config)
    biohama.start()
    
    try:
        # 훈련 실행
        training_results = []
        for epoch in range(epochs):
            click.echo(f"에포크 {epoch + 1}/{epochs} 진행 중...")
            result = biohama.train(training_data)
            training_results.append(result)
            
            # 진행률 표시
            progress = (epoch + 1) / epochs * 100
            click.echo(f"진행률: {progress:.1f}%")
        
        # 최종 결과
        final_result = {
            'epochs': epochs,
            'total_samples': sum(r['training_samples'] for r in training_results),
            'final_statistics': training_results[-1]['final_statistics'],
            'training_history': training_results
        }
        
        # 결과 출력
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ 훈련 결과가 {output}에 저장되었습니다.")
        else:
            click.echo("✅ 훈련 완료!")
            click.echo(f"총 샘플 수: {final_result['total_samples']}")
            click.echo("최종 통계:")
            click.echo(json.dumps(final_result['final_statistics'], indent=2, ensure_ascii=False))
            
    except Exception as e:
        click.echo(f"❌ 훈련 실패: {e}")
        raise click.Abort()
    finally:
        biohama.stop()


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='설정 파일 경로')
@click.option('--checkpoint', '-cp', type=click.Path(), help='체크포인트 파일 경로')
@click.option('--output', '-o', type=click.Path(), help='통계 출력 파일 경로')
def stats(config: str, checkpoint: Optional[str], output: Optional[str]):
    """시스템 통계를 확인합니다."""
    
    click.echo("BioHama 시스템 통계 확인 중...")
    
    # 설정 로드
    with open(config, 'r', encoding='utf-8') as f:
        if config.endswith('.yaml') or config.endswith('.yml'):
            system_config = yaml.safe_load(f)
        else:
            system_config = json.load(f)
    
    # 시스템 초기화
    biohama = BioHamaSystem(system_config)
    
    # 체크포인트 로드 (있는 경우)
    if checkpoint:
        biohama.load_checkpoint(checkpoint)
        click.echo(f"체크포인트 로드됨: {checkpoint}")
    
    biohama.start()
    
    try:
        # 통계 수집
        stats = biohama.get_system_statistics()
        
        # 결과 출력
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ 시스템 통계가 {output}에 저장되었습니다.")
        else:
            click.echo("📊 시스템 통계:")
            click.echo(json.dumps(stats, indent=2, ensure_ascii=False))
            
    except Exception as e:
        click.echo(f"❌ 통계 수집 실패: {e}")
        raise click.Abort()
    finally:
        biohama.stop()


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='설정 파일 경로')
@click.option('--checkpoint', '-cp', type=click.Path(), required=True, help='체크포인트 저장 경로')
def save(config: str, checkpoint: str):
    """시스템 체크포인트를 저장합니다."""
    
    click.echo("BioHama 시스템 체크포인트 저장 중...")
    
    # 설정 로드
    with open(config, 'r', encoding='utf-8') as f:
        if config.endswith('.yaml') or config.endswith('.yml'):
            system_config = yaml.safe_load(f)
        else:
            system_config = json.load(f)
    
    # 시스템 초기화
    biohama = BioHamaSystem(system_config)
    biohama.start()
    
    try:
        # 체크포인트 저장
        biohama.save_checkpoint(checkpoint)
        click.echo(f"✅ 체크포인트가 {checkpoint}에 저장되었습니다.")
        
    except Exception as e:
        click.echo(f"❌ 체크포인트 저장 실패: {e}")
        raise click.Abort()
    finally:
        biohama.stop()


@main.command()
def demo():
    """데모를 실행합니다."""
    
    click.echo("BioHama 시스템 데모 실행 중...")
    
    try:
        # 데모 설정
        config = {
            'device': 'cpu',
            'meta_router': {
                'input_dim': 128,
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
                'metacognitive_dim': 128,
                'max_history_length': 1000
            },
            'working_memory': {
                'capacity': 256,
                'chunk_size': 64,
                'decay_rate': 0.1,
                'consolidation_threshold': 0.7
            },
            'decision_engine': {
                'decision_dim': 256,
                'num_options': 10,
                'exploration_rate': 0.1
            },
            'attention_control': {
                'attention_dim': 128,
                'num_heads': 8,
                'dropout': 0.1
            },
            'message_passing': {
                'message_dim': 128,
                'max_message_length': 1000,
                'message_ttl': 10,
                'input_dim': 128,
                'output_dim': 128
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
                'replay_buffer_size': 10000,
                'batch_size': 64,
                'state_dim': 128,
                'action_dim': 64
            }
        }
        
        # 시스템 초기화
        biohama = BioHamaSystem(config)
        biohama.start()
        
        # 데모 입력들
        demo_inputs = [
            {'text': '안녕하세요, BioHama 시스템입니다.', 'task_type': 'greeting'},
            {'text': '간단한 계산을 해주세요: 2 + 3 = ?', 'task_type': 'calculation'},
            {'text': '오늘 날씨는 어떤가요?', 'task_type': 'weather_inquiry'}
        ]
        
        click.echo("🧠 BioHama 시스템 데모")
        click.echo("=" * 50)
        
        for i, inputs in enumerate(demo_inputs, 1):
            click.echo(f"\n입력 {i}: {inputs['text']}")
            
            result = biohama.process_input(inputs)
            
            click.echo(f"선택된 모듈: {result.get('selected_module', 'N/A')}")
            click.echo(f"신뢰도: {result.get('routing_confidence', 0.0):.3f}")
            
            if 'error' in result:
                click.echo(f"오류: {result['error']}")
        
        # 시스템 통계
        stats = biohama.get_system_statistics()
        click.echo(f"\n📊 시스템 통계:")
        click.echo(f"시스템 상태: {stats['system_status']['initialized']}")
        click.echo(f"작업 메모리 사용률: {stats['working_memory']['utilization']:.2f}")
        
        biohama.stop()
        click.echo("\n✅ 데모 완료!")
        
    except Exception as e:
        click.echo(f"❌ 데모 실행 실패: {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()
