import json
import subprocess
from pathlib import Path
from dataclasses import asdict
from typing import List

from .gasp import GASPResult
from .geometry import Item, Knapsack

# Assumes the jar is at gasp_java/target/gasp-solver.jar relative to project root
JAR_PATH = Path(__file__).resolve().parents[1] / "gasp_java" / "target" / "gasp-solver.jar"

class MockPacking:
    def __init__(self, used_volume):
        self.used_volume = used_volume

class JavaGASP:
    """A drop-in replacement for AdaptiveGASP that delegates to the Java engine."""
    def __init__(self, items: List[Item], knapsack: Knapsack, params=None, solver="gasp"):
        self.items = items
        self.knapsack = knapsack
        self.params = params
        self.solver = solver
    
    def run(self) -> GASPResult:
        if not JAR_PATH.exists():
            raise FileNotFoundError(f"Java solver JAR not found at {JAR_PATH}. Run 'mvn clean package' in gasp_java first.")
            
        # 1. Build JSON input
        input_data = {
            "solver": self.solver,
            "knapsack": {
                "w": self.knapsack.W,
                "d": self.knapsack.D,
                "h": self.knapsack.H
            },
            "items": [
                {
                    "idx": item.idx,
                    "w": item.w,
                    "d": item.d,
                    "h": item.h,
                    "profit": item.profit
                }
                for item in self.items
            ],
            "params": asdict(self.params) if self.params else {}
        }
        
        # 2. Call Java JAR via subprocess (stdin/stdout)
        proc = subprocess.run(
            ["java", "-jar", str(JAR_PATH)],
            input=json.dumps(input_data).encode("utf-8"),
            capture_output=True
        )
        
        if proc.returncode != 0:
            raise RuntimeError(f"Java solver failed with exit code {proc.returncode}:\n{proc.stderr.decode('utf-8')}")
            
        # 3. Parse JSON output
        output_data = json.loads(proc.stdout.decode("utf-8"))
        
        # 4. Construct a mock GASPResult to satisfy runner.py
        res = GASPResult(
            best_packing=MockPacking(output_data["volume"]),
            best_profit=output_data["profit"],
            iterations=output_data["iterations"],
            elapsed=output_data["elapsed"],
            history=[],
            seed_volume=None,
            pre_layout_volume=None
        )
        
        return res
