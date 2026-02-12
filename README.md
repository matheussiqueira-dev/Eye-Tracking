# Eye Tracking em Tempo Real (Webcam Comum)

Sistema de eye tracking em tempo real, sem hardware proprietario, com:

- deteccao facial e landmarks de iris (MediaPipe Face Mesh + Iris)
- estimativa de direcao do olhar com compensacao de pose da cabeca
- calibracao linear (9 pontos)
- estabilizacao temporal (Kalman + One Euro + rejeicao de outliers)
- geracao de heatmap de atencao visual
- exportacao de eventos em NDJSON para analytics

## Requisitos

- Python 3.10-3.12 (MediaPipe costuma ser mais estavel nessas versoes)
- Webcam

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execucao

```bash
python run.py --camera-id 0 --width 1280 --height 720 --target-fps 120
```

Parametros uteis:

- `--process-every-n 2` reduz custo de inferencia (mais FPS, menos acuracia temporal)
- `--log-events data/events.ndjson` salva stream de gaze para analise posterior
- `--no-debug` remove overlays de diagnostico
- `--no-heatmap` remove renderizacao do heatmap

## Atalhos em runtime

- `C`: iniciar calibracao (9 pontos)
- `H`: ligar/desligar heatmap
- `D`: ligar/desligar debug overlay
- `R`: resetar calibracao + heatmap
- `S`: salvar screenshot com heatmap
- `Q` ou `ESC`: sair

## Pipeline

1. Captura assincrona da webcam.
2. Deteccao de face + landmarks (face/olhos/iris).
3. Vetorizacao dos olhos (`horizontal`, `vertical`) e qualidade por olho.
4. Estimativa de pose da cabeca (`solvePnP`).
5. Fusão binocular + fallback de redundancia.
6. Mapeamento para coordenadas normalizadas de tela (`[0,1]x[0,1]`).
7. Estabilizacao temporal e rejeicao de ruido.
8. Acumulacao de heatmap e emissao de eventos.

## Estrutura

```text
run.py
src/eye_tracking/
  calibration.py
  capture.py
  config.py
  filters.py
  gaze.py
  heatmap.py
  landmarks.py
  runner.py
```

## Observacoes de performance

- Em CPU comum, 60+ FPS e realista; com tuning de ROI/inferencia e GPU e possivel subir bastante.
- `process_every_n` e resolucao de captura sao os principais knobs.
- Para producao, recomenda-se migrar o loop critico para C++/Rust ou ONNX/TensorRT.
