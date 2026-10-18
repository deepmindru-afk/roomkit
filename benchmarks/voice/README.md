# Voice frame cost

How much CPU RoomKit spends on one inbound 20 ms voice frame, once a
transport has decoded it. Frames enter where a voice backend's audio
callback hands them over (`MockVoiceBackend.simulate_audio_received`), on
N sessions with one room each, so the figure covers the audio pipeline,
the VAD, the STT hand-off and each turn's inbound path. RTP, codecs and
models are excluded: add the RTP figure from a transport benchmark to get
the cost of a whole call.

```bash
# Energy VAD, batch mock STT, 100 calls, 10 s, three runs, pinned to CPU 2.
uv run python -m benchmarks.voice --calls 100 --seconds 10 --cpu 2

# No VAD, streaming mock STT.
uv run python -m benchmarks.voice --vad none --stt stream
```

The audio is 8 kHz (a G.711 call): one second of a 440 Hz tone, then one
second of silence, so the energy VAD opens and closes one turn per call
every two seconds. That is far more turns than real speech, which makes
the per-turn share of the cost an upper bound. Each line reports µs per
frame, the calls one core would carry for this layer alone, and the STT
requests (batch transcriptions plus streams opened), which show that the
same turns were detected when comparing two runs.

Compare runs from one machine, quiet, pinned to one performance core.
