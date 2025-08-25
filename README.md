# SecurityThreat_Multi-Camera

Prototype of a multi-camera security system using a simple handover network.

## Installation

```
pip install -r requirements.txt
```

Both OpenCV and PyTorch are optional; when they are not available the demo
falls back to blank frames and random feature vectors so it can still be
executed in lightweight environments.

For a bare‑bones run with stubbed vision and embeddings you only need:

```
pip install numpy pyzmq
```

## Usage

Provide camera indices or RTSP URLs as positional arguments. If no arguments
are supplied a single dummy source is used.

```
python orchestrator.py 0 1
```

Use `Ctrl+C` to stop the system.
