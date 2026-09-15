"""CPU f32 recurrent cells with explicit caller-owned state.

Weights: [gates*H,I], [gates*H,H]; independent optional biases [gates*H].
GRU gates reset/update/new use reset-after semantics; LSTM gates are i/f/g/o.
Unroll is time-major [T,B,I], lengths has B integer entries; reset is flat T*B.
Padding produces zero outputs and retains state. Reset restores initial state.
truncate=k detaches incoming state at t=k,2k,...; state.detach() cuts chunks.
No fused scan, graph replay, CUDA execution, or bounded-memory promise.
"""
from ._native import RecurrentState, UnrollOutput, rnn_cell, gru_cell, lstm_cell, unroll
