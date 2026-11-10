"""Deterministic normal-mode Graphwar mechanics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from graphwar_rl.equations.ast import EquationNode, evaluate_ast
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.types import (
    EquationAction,
    GameState,
    ShotResult,
    SoldierState,
    Team,
    Vec2,
)


@dataclass(slots=True)
class _Soldier:
    soldier_id: int
    player_id: int
    team: Team
    x_coordinate: float
    y_coordinate: float
    alive: bool = True


class PythonGraphwarSimulator:
    """A deterministic, headless implementation of Graphwar normal mode."""

    plane_width = 770
    plane_height = 450
    game_width = 50.0
    soldier_radius = 7
    explosion_radius = 12
    step_size = 0.01
    max_steps = 20_000
    max_step_distance_squared = 0.001
    min_x_step_distance = 0.00001
    angle_error = math.pi / 360.0
    max_angle_loops = 100

    def __init__(self, max_equation_tokens: int, max_trajectory_points: int = 512) -> None:
        self.grammar = EquationGrammar(max_equation_tokens)
        self.max_trajectory_points = max_trajectory_points
        self._random = random.Random()
        self._scenario: ScenarioConfig | None = None
        self._scenario_seed = 0
        self._terrain = torch.zeros(
            (self.plane_height, self.plane_width),
            dtype=torch.bool,
        )
        self._obstacle_circles: list[tuple[int, int, int]] = []
        self._terrain_explosions: list[tuple[int, int, int]] = []
        self._soldiers: list[_Soldier] = []
        self._player_ids: list[int] = []
        self._player_teams: dict[int, Team] = {}
        self._current_soldier_by_player: dict[int, int] = {}
        self._current_player_position = 0
        self._turn_index = 0
        self._terminated = False
        self._truncated = False
        self._winning_team: Team | None = None
        self._closed = False

    def reset(self, seed: int, scenario: ScenarioConfig) -> GameState:
        """Reset to a deterministic generated scenario."""

        if self._closed:
            raise RuntimeError("Simulator is closed")
        scenario.validate()
        self._scenario = scenario
        self._scenario_seed = int(seed)
        self._random.seed(self._scenario_seed)
        self._terrain.zero_()
        self._obstacle_circles.clear()
        self._terrain_explosions.clear()
        self._soldiers.clear()
        self._player_ids.clear()
        self._player_teams.clear()
        self._current_soldier_by_player.clear()
        self._turn_index = 0
        self._terminated = False
        self._truncated = False
        self._winning_team = None

        self._generate_obstacles(scenario.obstacle_count)
        self._generate_players(scenario)
        starting_player = next(
            player_id
            for player_id in self._player_ids
            if self._player_teams[player_id] is scenario.starting_team
        )
        self._current_player_position = self._player_ids.index(starting_player)
        return self._state()

    def step(self, action: EquationAction) -> tuple[GameState, ShotResult]:
        """Simulate one equation without mutating state on invalid input."""

        if self._closed:
            raise RuntimeError("Simulator is closed")
        if self._scenario is None:
            raise RuntimeError("reset must be called before step")
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot step a finished episode")

        try:
            equation = self.grammar.decode(action, self._scenario.mode)
        except ValueError as error:
            result = self._invalid_result("invalid_equation", str(error))
            return self._state(), result

        trace = self._trace_equation(equation)
        if trace is None:
            result = self._invalid_result(
                "non_finite_equation",
                "Equation is undefined at the firing position",
            )
            return self._state(), result

        trajectory, hit_ids, closest_distance, collision_point = trace
        shooter = self._current_shooter()
        enemy_hits = tuple(
            soldier_id
            for soldier_id in hit_ids
            if self._soldier_by_id(soldier_id).team is not shooter.team
        )
        ally_hits = tuple(
            soldier_id
            for soldier_id in hit_ids
            if self._soldier_by_id(soldier_id).team is shooter.team
        )
        for soldier_id in hit_ids:
            self._soldier_by_id(soldier_id).alive = False

        if collision_point is not None:
            self._explode_terrain(*collision_point)

        self._turn_index += 1
        alive_teams = {
            soldier.team
            for soldier in self._soldiers
            if soldier.alive
        }
        if len(alive_teams) == 1:
            self._terminated = True
            self._winning_team = next(iter(alive_teams))
            terminal_reason = "team_eliminated"
        elif self._turn_index >= self._scenario.max_turns:
            self._truncated = True
            terminal_reason = "max_turns"
        else:
            self._advance_turn()
            terminal_reason = None

        result = ShotResult(
            valid=True,
            trajectory=trajectory,
            enemy_hits=enemy_hits,
            ally_hits=ally_hits,
            closest_enemy_distance=closest_distance,
            terminated=self._terminated,
            truncated=self._truncated,
            winning_team=self._winning_team,
            terminal_reason=terminal_reason,
        )
        return self._state(), result

    def close(self) -> None:
        """Release simulator resources."""

        self._closed = True
        self._scenario = None
        self._soldiers.clear()

    def snapshot(self) -> dict[str, Any]:
        """Serialize the full live episode state."""

        if self._scenario is None:
            raise RuntimeError("Cannot snapshot before reset")
        return {
            "format_version": 1,
            "scenario": self._scenario,
            "scenario_seed": self._scenario_seed,
            "random_state": self._random.getstate(),
            "terrain": self._terrain.clone(),
            "obstacle_circles": tuple(self._obstacle_circles),
            "terrain_explosions": tuple(self._terrain_explosions),
            "soldiers": tuple(
                {
                    "soldier_id": soldier.soldier_id,
                    "player_id": soldier.player_id,
                    "team": int(soldier.team),
                    "x_coordinate": soldier.x_coordinate,
                    "y_coordinate": soldier.y_coordinate,
                    "alive": soldier.alive,
                }
                for soldier in self._soldiers
            ),
            "player_ids": tuple(self._player_ids),
            "player_teams": {
                player_id: int(team)
                for player_id, team in self._player_teams.items()
            },
            "current_soldier_by_player": dict(self._current_soldier_by_player),
            "current_player_position": self._current_player_position,
            "turn_index": self._turn_index,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "winning_team": (
                None if self._winning_team is None else int(self._winning_team)
            ),
        }

    def restore(self, snapshot: dict[str, Any]) -> GameState:
        """Restore a full live episode state."""

        if self._closed:
            raise RuntimeError("Simulator is closed")
        if snapshot.get("format_version") != 1:
            raise ValueError("Unsupported simulator snapshot format")
        scenario = snapshot.get("scenario")
        if not isinstance(scenario, ScenarioConfig):
            raise ValueError("Simulator snapshot has an invalid scenario")
        scenario.validate()
        terrain = snapshot.get("terrain")
        if not isinstance(terrain, Tensor) or terrain.shape != (
            self.plane_height,
            self.plane_width,
        ):
            raise ValueError("Simulator snapshot has an invalid terrain tensor")

        self._scenario = scenario
        self._scenario_seed = int(snapshot["scenario_seed"])
        self._random.setstate(snapshot["random_state"])
        self._terrain = terrain.clone().to(dtype=torch.bool, device="cpu")
        self._obstacle_circles = [
            tuple(int(value) for value in circle)
            for circle in snapshot["obstacle_circles"]
        ]
        self._terrain_explosions = [
            tuple(int(value) for value in circle)
            for circle in snapshot["terrain_explosions"]
        ]
        self._soldiers = [
            _Soldier(
                soldier_id=int(values["soldier_id"]),
                player_id=int(values["player_id"]),
                team=Team(int(values["team"])),
                x_coordinate=float(values["x_coordinate"]),
                y_coordinate=float(values["y_coordinate"]),
                alive=bool(values["alive"]),
            )
            for values in snapshot["soldiers"]
        ]
        self._player_ids = [int(value) for value in snapshot["player_ids"]]
        self._player_teams = {
            int(player_id): Team(int(team))
            for player_id, team in snapshot["player_teams"].items()
        }
        self._current_soldier_by_player = {
            int(player_id): int(soldier_id)
            for player_id, soldier_id in snapshot[
                "current_soldier_by_player"
            ].items()
        }
        self._current_player_position = int(snapshot["current_player_position"])
        self._turn_index = int(snapshot["turn_index"])
        self._terminated = bool(snapshot["terminated"])
        self._truncated = bool(snapshot["truncated"])
        winning_team = snapshot["winning_team"]
        self._winning_team = None if winning_team is None else Team(int(winning_team))
        return self._state()

    def get_rng_state(self) -> object:
        """Return the simulator RNG state for checkpointing."""

        return self._random.getstate()

    def set_rng_state(self, state: object) -> None:
        """Restore the simulator RNG state from a checkpoint."""

        self._random.setstate(state)

    def _invalid_result(self, code: str, message: str) -> ShotResult:
        return ShotResult(
            valid=False,
            trajectory=None,
            enemy_hits=(),
            ally_hits=(),
            closest_enemy_distance=None,
            terminated=False,
            truncated=False,
            winning_team=None,
            error_code=code,
            error=message,
        )

    def _generate_obstacles(self, obstacle_count: int) -> None:
        vertical_indices, horizontal_indices = torch.meshgrid(
            torch.arange(self.plane_height),
            torch.arange(self.plane_width),
            indexing="ij",
        )
        for _ in range(obstacle_count):
            radius = self._random.randint(18, 48)
            center_x = self._random.randint(150, self.plane_width - 150)
            center_y = self._random.randint(70, self.plane_height - 70)
            self._obstacle_circles.append((center_x, center_y, radius))
            circle = (
                (horizontal_indices - center_x).square()
                + (vertical_indices - center_y).square()
                <= radius * radius
            )
            self._terrain.logical_or_(circle)

    def _generate_players(self, scenario: ScenarioConfig) -> None:
        total_players = scenario.players_per_team * 2
        for player_id in range(total_players):
            team = Team.ONE if player_id < scenario.players_per_team else Team.TWO
            self._player_ids.append(player_id)
            self._player_teams[player_id] = team
            first_soldier_id = len(self._soldiers)
            self._current_soldier_by_player[player_id] = first_soldier_id
            for _ in range(scenario.soldiers_per_player):
                position = self._sample_soldier_position(team)
                self._soldiers.append(
                    _Soldier(
                        soldier_id=len(self._soldiers),
                        player_id=player_id,
                        team=team,
                        x_coordinate=position[0],
                        y_coordinate=position[1],
                    )
                )

    def _sample_soldier_position(self, team: Team) -> tuple[float, float]:
        if team is Team.ONE:
            minimum_x, maximum_x = 40, 310
        else:
            minimum_x, maximum_x = self.plane_width - 310, self.plane_width - 40
        for _ in range(2_000):
            x_coordinate = self._random.randint(minimum_x, maximum_x)
            y_coordinate = self._random.randint(35, self.plane_height - 35)
            if self._terrain_near(x_coordinate, y_coordinate, self.soldier_radius + 2):
                continue
            if any(
                math.hypot(
                    soldier.x_coordinate - x_coordinate,
                    soldier.y_coordinate - y_coordinate,
                )
                < self.soldier_radius * 3
                for soldier in self._soldiers
            ):
                continue
            return float(x_coordinate), float(y_coordinate)
        raise RuntimeError("Could not place soldiers for the requested scenario")

    def _terrain_near(self, x_coordinate: int, y_coordinate: int, radius: int) -> bool:
        minimum_x = max(0, x_coordinate - radius)
        maximum_x = min(self.plane_width, x_coordinate + radius + 1)
        minimum_y = max(0, y_coordinate - radius)
        maximum_y = min(self.plane_height, y_coordinate + radius + 1)
        return bool(self._terrain[minimum_y:maximum_y, minimum_x:maximum_x].any())

    def _current_shooter(self) -> _Soldier:
        player_id = self._player_ids[self._current_player_position]
        soldier_id = self._current_soldier_by_player[player_id]
        shooter = self._soldier_by_id(soldier_id)
        if not shooter.alive:
            raise RuntimeError("Current shooter is not alive")
        return shooter

    def _soldier_by_id(self, soldier_id: int) -> _Soldier:
        return self._soldiers[soldier_id]

    def _trace_equation(
        self,
        equation: EquationNode,
    ) -> tuple[Tensor, tuple[int, ...], float | None, tuple[int, int] | None] | None:
        shooter = self._current_shooter()
        inverted = shooter.team is Team.TWO
        start_pixel_x = (
            self.plane_width - shooter.x_coordinate
            if inverted
            else shooter.x_coordinate
        )
        start_game_x = (
            self.game_width * (start_pixel_x - self.plane_width / 2)
            / self.plane_width
        )
        start_game_y = (
            self.game_width * (-shooter.y_coordinate + self.plane_height / 2)
            / self.plane_width
        )
        game_radius = self.game_width * self.soldier_radius / self.plane_width
        fire_angle = self._start_angle(equation, start_game_x, game_radius)
        if not math.isfinite(fire_angle):
            return None

        current_game_x = start_game_x + game_radius * math.cos(fire_angle)
        current_game_y = start_game_y + game_radius * math.sin(fire_angle)
        function_at_start = evaluate_ast(equation, current_game_x)
        if not math.isfinite(function_at_start):
            return None
        offset = -function_at_start + current_game_y

        points = [
            self._game_to_pixel(current_game_x, current_game_y, inverted)
        ]
        hit_ids: list[int] = []
        closest_enemy_distance: float | None = None
        collision_point: tuple[int, int] | None = None

        for _ in range(1, self.max_steps):
            candidate_step = self.step_size
            next_game_x = current_game_x + candidate_step
            next_game_y = evaluate_ast(equation, next_game_x) + offset
            if not math.isfinite(next_game_y):
                break

            while (
                (next_game_x - current_game_x) ** 2
                + (next_game_y - current_game_y) ** 2
                > self.max_step_distance_squared
            ):
                if next_game_x - current_game_x <= self.min_x_step_distance:
                    next_game_x = current_game_x
                    break
                candidate_step /= 2.0
                next_game_x = current_game_x + candidate_step
                next_game_y = evaluate_ast(equation, next_game_x) + offset
                if not math.isfinite(next_game_y):
                    break

            if next_game_x == current_game_x or not math.isfinite(next_game_y):
                break
            pixel_x, pixel_y = self._game_to_pixel(next_game_x, next_game_y, inverted)
            points.append((pixel_x, pixel_y))

            for soldier in self._soldiers:
                if not soldier.alive or soldier.soldier_id == shooter.soldier_id:
                    continue
                distance = math.hypot(
                    soldier.x_coordinate - pixel_x,
                    soldier.y_coordinate - pixel_y,
                )
                if soldier.team is not shooter.team:
                    closest_enemy_distance = (
                        distance
                        if closest_enemy_distance is None
                        else min(closest_enemy_distance, distance)
                    )
                if distance < self.soldier_radius and soldier.soldier_id not in hit_ids:
                    hit_ids.append(soldier.soldier_id)

            integer_x = int(pixel_x)
            integer_y = int(pixel_y)
            if (
                integer_x < 0
                or integer_x >= self.plane_width
                or integer_y < 0
                or integer_y >= self.plane_height
            ):
                break
            if self._terrain[integer_y, integer_x]:
                collision_point = (integer_x, integer_y)
                break

            current_game_x = next_game_x
            current_game_y = next_game_y

        trajectory = torch.tensor(points, dtype=torch.float32)
        if trajectory.shape[0] > self.max_trajectory_points:
            sample_indices = torch.linspace(
                0,
                trajectory.shape[0] - 1,
                self.max_trajectory_points,
            ).round().to(dtype=torch.long)
            trajectory = trajectory[sample_indices]
        return trajectory, tuple(hit_ids), closest_enemy_distance, collision_point

    def _start_angle(
        self,
        equation: EquationNode,
        start_game_x: float,
        game_radius: float,
    ) -> float:
        initial = evaluate_ast(equation, start_game_x)
        following = evaluate_ast(equation, start_game_x + self.step_size)
        if not math.isfinite(initial) or not math.isfinite(following):
            return math.nan
        angle = math.atan((following - initial) / self.step_size)
        error = math.inf
        loop_count = 0
        while error > self.angle_error and loop_count < self.max_angle_loops:
            final_x = start_game_x + game_radius * math.cos(angle)
            current = evaluate_ast(equation, final_x)
            following = evaluate_ast(equation, final_x + self.step_size)
            if not math.isfinite(current) or not math.isfinite(following):
                return math.nan
            new_angle = math.atan((following - current) / self.step_size)
            error = abs(new_angle - angle)
            angle = new_angle
            loop_count += 1
        return angle

    def _game_to_pixel(
        self,
        game_x: float,
        game_y: float,
        inverted: bool,
    ) -> tuple[float, float]:
        pixel_x = (
            self.plane_width * game_x / self.game_width
            + self.plane_width / 2
        )
        pixel_y = (
            -self.plane_width * game_y / self.game_width
            + self.plane_height / 2
        )
        if inverted:
            pixel_x = self.plane_width - pixel_x
        return pixel_x, pixel_y

    def _explode_terrain(self, center_x: int, center_y: int) -> None:
        self._terrain_explosions.append(
            (center_x, center_y, self.explosion_radius)
        )
        minimum_x = max(0, center_x - self.explosion_radius)
        maximum_x = min(self.plane_width, center_x + self.explosion_radius + 1)
        minimum_y = max(0, center_y - self.explosion_radius)
        maximum_y = min(self.plane_height, center_y + self.explosion_radius + 1)
        vertical_indices, horizontal_indices = torch.meshgrid(
            torch.arange(minimum_y, maximum_y),
            torch.arange(minimum_x, maximum_x),
            indexing="ij",
        )
        circle = (
            (horizontal_indices - center_x).square()
            + (vertical_indices - center_y).square()
            <= self.explosion_radius * self.explosion_radius
        )
        region = self._terrain[minimum_y:maximum_y, minimum_x:maximum_x]
        region.logical_and_(~circle)

    def _advance_turn(self) -> None:
        for player_offset in range(1, len(self._player_ids) + 1):
            candidate_position = (
                self._current_player_position + player_offset
            ) % len(self._player_ids)
            player_id = self._player_ids[candidate_position]
            player_soldiers = [
                soldier
                for soldier in self._soldiers
                if soldier.player_id == player_id
            ]
            if not any(soldier.alive for soldier in player_soldiers):
                continue
            current_id = self._current_soldier_by_player[player_id]
            current_index = next(
                index
                for index, soldier in enumerate(player_soldiers)
                if soldier.soldier_id == current_id
            )
            for soldier_offset in range(1, len(player_soldiers) + 1):
                candidate = player_soldiers[
                    (current_index + soldier_offset) % len(player_soldiers)
                ]
                if candidate.alive:
                    self._current_soldier_by_player[player_id] = candidate.soldier_id
                    self._current_player_position = candidate_position
                    return
        raise RuntimeError("No alive player is available for the next turn")

    def _state(self) -> GameState:
        assert self._scenario is not None
        shooter = self._current_shooter()
        terrain = functional.interpolate(
            self._terrain.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(
                self._scenario.observation_height,
                self._scenario.observation_width,
            ),
            mode="nearest",
        ).squeeze(0)
        soldiers = tuple(
            SoldierState(
                soldier_id=soldier.soldier_id,
                player_id=soldier.player_id,
                team=soldier.team,
                position=Vec2(soldier.x_coordinate, soldier.y_coordinate),
                alive=soldier.alive,
                is_current_shooter=soldier.soldier_id == shooter.soldier_id,
            )
            for soldier in self._soldiers
        )
        return GameState(
            terrain=terrain,
            soldiers=soldiers,
            current_player_id=shooter.player_id,
            current_soldier_id=shooter.soldier_id,
            mode=self._scenario.mode,
            turn_index=self._turn_index,
            max_turns=self._scenario.max_turns,
            scenario_seed=self._scenario_seed,
            terminated=self._terminated,
            truncated=self._truncated,
            winning_team=self._winning_team,
        )
