"""Scan rustc --emit=asm output (any target) for AVX-encoded instructions.
Functions are delimited by top-level labels; reports each function holding
VEX/EVEX mnemonics and every direct, symbol-named call or jump into such a
function. Indirect calls (function pointers, vtables) cannot be attributed;
those paths need separate review, as for the GEMM kernel table in gemm.rs.

usage: python3 vexscan_asm.py file.s
"""
import re
import sys
from collections import defaultdict

label_re = re.compile(r'^("?[^\s.$"][^:]*"?):\s*(#.*)?$')
insn_re = re.compile(r'^\s+(v[a-z0-9]+)\s')
call_re = re.compile(r'^\s+(call[a-z]*|jmp)\s+("?[^\s"]+"?)')

vex = defaultdict(set)
callers = defaultdict(set)
cur = None
for line in open(sys.argv[1], errors='replace'):
    m = label_re.match(line)
    if m and not line.startswith('\t'):
        cur = m.group(1).strip('"')
        continue
    if cur is None:
        continue
    i = insn_re.match(line)
    if i and i.group(1) not in ('verr', 'verw'):
        vex[cur].add(i.group(1))
    c = call_re.match(line)
    if c:
        callers[c.group(2).strip('"')].add(cur)

for f in sorted(vex):
    print(f"VEX FUNC: {f}  insns={sorted(vex[f])[:8]}")
    for c in sorted(callers.get(f, [])):
        print(f"    called from: {c}")
print(f"{len(vex)} functions contain VEX/EVEX instructions")
