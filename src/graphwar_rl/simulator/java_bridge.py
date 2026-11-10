"""Subprocess bridge to the Java normal-mode mechanics oracle."""

from __future__ import annotations

import math
import select
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from graphwar_rl.equations.ast import EquationNode, ast_to_tokens
from graphwar_rl.simulator.python_sim import PythonGraphwarSimulator


class JavaGraphwarSimulator(PythonGraphwarSimulator):
    """Use Java for trajectories while retaining deterministic Python episodes."""

    def __init__(
        self,
        jar_path: Path,
        max_equation_tokens: int,
        max_trajectory_points: int = 512,
        timeout_seconds: float = 5.0,
    ) -> None:
        super().__init__(max_equation_tokens, max_trajectory_points)
        self.jar_path = jar_path
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        """Launch and validate the Java mechanics subprocess."""

        if self.process is not None and self.process.poll() is None:
            return
        if not self.jar_path.is_file():
            raise FileNotFoundError(
                f"Java simulator JAR not found: {self.jar_path}. "
                "Run java-sim/build.sh first."
            )
        self.process = subprocess.Popen(
            ["java", "-jar", str(self.jar_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        response = self._send_request("PING")
        if response != ["PONG"]:
            self.close()
            raise RuntimeError(f"Unexpected Java simulator handshake: {response}")

    def close(self) -> None:
        """Close the Java process and Python episode state."""

        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("CLOSE\n")
                    process.stdin.flush()
                process.wait(timeout=1.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        super().close()

    def _trace_equation(
        self,
        equation: EquationNode,
    ) -> tuple[Tensor, tuple[int, ...], float | None, tuple[int, int] | None] | None:
        if self.process is None or self.process.poll() is not None:
            self.start()
        tokens, constants = ast_to_tokens(equation)
        shooter = self._current_shooter()
        soldiers = ";".join(
            ",".join(
                [
                    str(soldier.soldier_id),
                    str(int(soldier.team)),
                    format(soldier.x_coordinate, ".17g"),
                    format(soldier.y_coordinate, ".17g"),
                    "1" if soldier.alive else "0",
                ]
            )
            for soldier in self._soldiers
        )
        obstacles = ";".join(
            f"{center_x},{center_y},{radius}"
            for center_x, center_y, radius in self._obstacle_circles
        )
        explosions = ";".join(
            f"{center_x},{center_y},{radius}"
            for center_x, center_y, radius in self._terrain_explosions
        )
        request = "\t".join(
            [
                "TRACE",
                ",".join(str(int(token)) for token in tokens),
                ",".join(format(value, ".17g") for value in constants),
                str(shooter.soldier_id),
                str(int(shooter.team)),
                format(shooter.x_coordinate, ".17g"),
                format(shooter.y_coordinate, ".17g"),
                soldiers,
                obstacles,
                explosions,
                str(self.max_trajectory_points),
            ]
        )
        response = self._send_request(request)
        if response[0] == "INVALID":
            return None
        if response[0] != "OK" or len(response) != 5:
            raise RuntimeError(f"Malformed Java simulator response: {response}")

        trajectory_values = []
        if response[1]:
            for point in response[1].split(";"):
                x_coordinate, y_coordinate = point.split(",", maxsplit=1)
                trajectory_values.append(
                    [float(x_coordinate), float(y_coordinate)]
                )
        trajectory = torch.tensor(trajectory_values, dtype=torch.float32)
        hit_ids = (
            tuple(int(value) for value in response[2].split(","))
            if response[2]
            else ()
        )
        closest_distance = (
            None if response[3] == "null" else float(response[3])
        )
        collision_point = None
        if response[4] != "null":
            x_coordinate, y_coordinate = response[4].split(",", maxsplit=1)
            collision_point = (int(x_coordinate), int(y_coordinate))
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise RuntimeError("Java simulator returned an invalid trajectory shape")
        if not torch.isfinite(trajectory).all():
            raise RuntimeError("Java simulator returned a non-finite trajectory")
        if closest_distance is not None and not math.isfinite(closest_distance):
            raise RuntimeError("Java simulator returned a non-finite distance")
        return trajectory, hit_ids, closest_distance, collision_point

    def _send_request(self, request: str) -> list[str]:
        """Send one line and read one bounded response."""

        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Java simulator process is not running")
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Java simulator pipes are unavailable")
        process.stdin.write(request + "\n")
        process.stdin.flush()
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            raise TimeoutError("Java simulator response timed out")
        response = process.stdout.readline()
        if response == "":
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read(2_000)
            raise RuntimeError(
                f"Java simulator exited before responding: {stderr.strip()}"
            )
        if len(response) > 5_000_000:
            raise RuntimeError("Java simulator response exceeded the safety limit")
        fields = response.rstrip("\n").split("\t")
        if fields[0] == "ERROR":
            message = fields[2] if len(fields) > 2 else "unknown Java error"
            raise RuntimeError(message)
        return fields
