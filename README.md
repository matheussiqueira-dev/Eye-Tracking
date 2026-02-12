# Eye Tracking em Tempo Real com Webcam

Sistema de Eye Tracking em tempo real para webcam comum, sem hardware proprietario, com foco em analise comportamental, alta taxa de frames e arquitetura extensivel para uso em produto.

## Visao Geral

Este projeto implementa um pipeline completo de rastreamento ocular com:

- deteccao facial e landmarks com MediaPipe
- modelagem vetorial de iris por olho
- estimativa de direcao do olhar com compensacao de pose de cabeca
- calibracao supervisionada (9 pontos)
- estabilizacao temporal com filtros de baixa latencia
- heatmap dinamico de atencao visual
- exportacao de eventos para analytics em NDJSON

## Demonstracao

![Demonstracao do sistema de Eye Tracking](assets/eye-tracking.gif)

## Principais Capacidades

- Processamento em tempo real com captura assincrona.
- Fusao binocular com peso por confianca.
- Fallback para pose da cabeca em cenarios de oclusao parcial.
- Rejeicao de outliers para reduzir jitter.
- Persistencia de eventos de gaze para analise offline e BI.

## Arquitetura

```text
Camera (Async) -> Face/Landmarks -> Gaze Estimation -> Stabilization -> Heatmap -> UI + Event Log
```

Componentes:

- `capture.py`: leitura assincrona da webcam com buffer reduzido.
- `landmarks.py`: deteccao de face e landmarks de olhos/iris.
- `gaze.py`: pose da cabeca (`solvePnP`) + estimativa de gaze.
- `filters.py`: One Euro + Kalman + controle de outlier.
- `calibration.py`: rotina de calibracao e regressao linear regularizada.
- `heatmap.py`: acumulador temporal com kernel gaussiano.
- `runner.py`: orquestracao do pipeline, debug overlay e atalhos.

## Requisitos

- Python 3.10 a 3.12
- Webcam USB ou integrada
- Windows, Linux ou macOS

Dependencias principais:

- `opencv-python`
- `mediapipe`
- `numpy`

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execucao

Comando recomendado:

```bash
python run.py --camera-id 0 --width 1280 --height 720 --target-fps 120
```

Exemplo para webcam externa (ex.: Brio 305):

```bash
python run.py --camera-id 1 --width 1920 --height 1080 --target-fps 120
```

## Parametros de Linha de Comando

- `--camera-id`: indice da camera no sistema.
- `--width`: largura de captura.
- `--height`: altura de captura.
- `--target-fps`: FPS solicitado ao driver da camera.
- `--process-every-n`: processa inferencia completa a cada N frames.
- `--no-debug`: desativa overlays de diagnostico.
- `--no-heatmap`: desativa sobreposicao do heatmap.
- `--log-events`: caminho para exportar eventos em NDJSON.

## Atalhos em Runtime

- `C`: iniciar calibracao (9 pontos).
- `H`: alternar heatmap.
- `D`: alternar debug overlay.
- `R`: resetar calibracao e heatmap.
- `S`: salvar screenshot com heatmap.
- `Q` ou `ESC`: encerrar aplicacao.

## Pipeline de Processamento

1. Captura assincrona do frame.
2. Deteccao de face e landmarks.
3. Extracao vetorial da posicao da iris por olho.
4. Estimativa de pose da cabeca.
5. Fusao binocular e estimativa do ponto de gaze.
6. Aplicacao de filtros temporais e controle de confianca.
7. Acumulacao no heatmap com decaimento temporal.
8. Emissao de evento estruturado para armazenamento.

## Estrutura do Projeto

```text
run.py
requirements.txt
README.md
src/
  eye_tracking/
    __init__.py
    calibration.py
    capture.py
    config.py
    filters.py
    gaze.py
    heatmap.py
    landmarks.py
    runner.py
```

## Desempenho

- Em CPU comum, 60+ FPS e viavel.
- Em hardware mais robusto e com tuning, pode subir significativamente.
- `process_every_n`, resolucao e qualidade da webcam sao os principais controles de performance.
- Para cenarios extremos de baixa latencia, recomenda-se acelerar inferencia com ONNX/TensorRT e migrar partes criticas para C++/Rust.

## Limitacoes Conhecidas

- Precisao depende de iluminacao, posicao da camera e etapa de calibracao.
- Oculos com reflexo intenso podem reduzir confianca de landmarks.
- Sem calibracao, o mapeamento e aproximado.

## Roadmap de Producao

- Servico de eventos em streaming (Kafka/Redpanda).
- Dashboard analitico com mapas de atencao por sessao.
- Testes automatizados de regressao de gaze.
- Observabilidade (latencia p95, uptime de tracking, confianca media).

## Licenca

Defina a licenca desejada para distribuicao e uso comercial antes de publicar para terceiros.

## Autoria

- Matheus Siqueira
- Site: `wwww.matheussiqueira.dev`
