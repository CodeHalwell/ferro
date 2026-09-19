import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'examples'))
from gnn_restart import prove, SEEDS
class GraphRestart(unittest.TestCase):
    def test_cpu_learning_and_fresh_process_restart(self):
        for seed in SEEDS:
            with self.subTest(seed=seed): prove(seed)
