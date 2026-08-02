"""Standalone visual task used to evaluate EVE through normal desktop perception.

The experiment deliberately has no programmatic connection to EVE and writes no
feedback or ground-truth files.  A human starts one stage from the command line;
EVE can only observe the resulting window and interact with it normally.
"""

import argparse
import random
import time
from dataclasses import dataclass

import pygame


WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
HEADER_HEIGHT = 100
FPS = 60

EPISODE_SECONDS = 60
COUNTDOWN_SECONDS = 6

BACKGROUND = (30, 30, 30)
HEADER_BACKGROUND = (18, 18, 18)
RED = (235, 60, 60)
BLUE = (60, 120, 240)
WHITE = (245, 245, 245)
YELLOW = (245, 220, 80)

CORRECT_SCORE = 1
WRONG_SCORE = -1
EMPTY_CLICK_SCORE = 0

TARGET_LIFETIME_MS = 2200
SPAWN_MIN_MS = 500
SPAWN_MAX_MS = 1200
INSTRUCTION_DURATION_MS = 15000


STAGES = {
    "stage1": {
        "title": "阶段一",
        "instruction": "点击所有红色圆形",
        "spawn_types": ["red_circle"],
        "allowed_types": ["red_circle"],
        "max_targets": 1,
        "change_instruction": False,
    },
    "stage2": {
        "title": "阶段二",
        "instruction": "点击所有出现的目标",
        "spawn_types": ["red_circle", "blue_triangle"],
        "allowed_types": ["red_circle", "blue_triangle"],
        "max_targets": 2,
        "change_instruction": False,
    },
    "stage3": {
        "title": "阶段三",
        "instruction": None,
        "spawn_types": ["red_circle", "blue_triangle"],
        "allowed_types": None,
        "max_targets": 2,
        "change_instruction": True,
    },
    "regression": {
        "title": "阶段一回归",
        "instruction": "点击所有红色圆形",
        "spawn_types": ["red_circle"],
        "allowed_types": ["red_circle"],
        "max_targets": 1,
        "change_instruction": False,
    },
}


@dataclass
class Target:
    target_type: str
    center: tuple[int, int]
    size: int
    expires_at_ms: int

    def draw(self, screen: pygame.Surface) -> None:
        if self.target_type == "red_circle":
            pygame.draw.circle(screen, RED, self.center, self.size // 2)
        elif self.target_type == "blue_triangle":
            pygame.draw.polygon(screen, BLUE, self.triangle_vertices())

    def triangle_vertices(self) -> list[tuple[int, int]]:
        x, y = self.center
        half = self.size // 2
        return [(x, y - half), (x - half, y + half), (x + half, y + half)]

    def contains(self, point: tuple[int, int]) -> bool:
        if self.target_type == "red_circle":
            return self.circle_contains(point)
        return self.triangle_contains(point)

    def circle_contains(self, point: tuple[int, int]) -> bool:
        px, py = point
        cx, cy = self.center
        radius = self.size / 2
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius**2

    def triangle_contains(self, point: tuple[int, int]) -> bool:
        a, b, c = self.triangle_vertices()

        def sign(
            p1: tuple[int, int],
            p2: tuple[int, int],
            p3: tuple[int, int],
        ) -> int:
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (
                p2[0] - p3[0]
            ) * (p1[1] - p3[1])

        d1 = sign(point, a, b)
        d2 = sign(point, b, c)
        d3 = sign(point, c, a)
        has_negative = d1 < 0 or d2 < 0 or d3 < 0
        has_positive = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_negative and has_positive)


class RedBlueGame:
    def __init__(self, stage_name: str, seed: int) -> None:
        self.stage = STAGES[stage_name]
        self.random = random.Random(seed)
        self.targets: list[Target] = []
        self.score = 0
        self.running = True
        self.current_instruction = self.stage["instruction"]
        self.allowed_types = list(self.stage["allowed_types"] or [])
        self.next_instruction_change_ms = 0
        self.next_spawn_ms = 0
        self.ends_at_ms = 0

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"Red Blue Experiment - {self.stage['title']}")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("Microsoft YaHei", 28)
        self.normal_font = pygame.font.SysFont("Microsoft YaHei", 22)

    def start(self) -> None:
        try:
            self.countdown()
            if not self.running:
                return

            started_at_ms = pygame.time.get_ticks()
            self.ends_at_ms = started_at_ms + EPISODE_SECONDS * 1000
            self.schedule_next_spawn()

            if self.stage["change_instruction"]:
                self.change_instruction()

            while self.running and pygame.time.get_ticks() < self.ends_at_ms:
                self.handle_events()
                if not self.running:
                    break
                self.update_instruction()
                self.expire_targets()
                self.spawn_if_needed()
                self.draw()
                self.clock.tick(FPS)

            self.show_final_screen()
        finally:
            pygame.quit()

    def countdown(self) -> None:
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            second_started = time.monotonic()
            while self.running and time.monotonic() - second_started < 1.0:
                self.handle_events()
                self.screen.fill(BACKGROUND)
                self.draw_center_text(str(remaining), YELLOW, WINDOW_HEIGHT // 2)
                pygame.display.flip()
                self.clock.tick(FPS)
            if not self.running:
                return

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_left_click(event.pos)

    def handle_left_click(self, position: tuple[int, int]) -> None:
        clicked_target = next(
            (target for target in reversed(self.targets) if target.contains(position)),
            None,
        )
        if clicked_target is None:
            self.score += EMPTY_CLICK_SCORE
            return

        if clicked_target.target_type in self.allowed_types:
            self.score += CORRECT_SCORE
        else:
            self.score += WRONG_SCORE
        self.targets.remove(clicked_target)

    def update_instruction(self) -> None:
        if self.stage["change_instruction"] and (
            pygame.time.get_ticks() >= self.next_instruction_change_ms
        ):
            self.change_instruction()

    def change_instruction(self) -> None:
        options = [
            ("只点击红色圆形", ["red_circle"]),
            ("只点击蓝色三角形", ["blue_triangle"]),
        ]
        available = [
            option for option in options if option[0] != self.current_instruction
        ]
        instruction, allowed = self.random.choice(available or options)
        self.current_instruction = instruction
        self.allowed_types = list(allowed)
        self.next_instruction_change_ms = (
            pygame.time.get_ticks() + INSTRUCTION_DURATION_MS
        )

    def schedule_next_spawn(self) -> None:
        delay_ms = self.random.randint(SPAWN_MIN_MS, SPAWN_MAX_MS)
        self.next_spawn_ms = pygame.time.get_ticks() + delay_ms

    def spawn_if_needed(self) -> None:
        if pygame.time.get_ticks() < self.next_spawn_ms:
            return
        if len(self.targets) >= self.stage["max_targets"]:
            return
        self.spawn_target()
        self.schedule_next_spawn()

    def spawn_target(self) -> None:
        target_type = self.random.choice(self.stage["spawn_types"])
        size = (
            self.random.randint(80, 120)
            if target_type == "red_circle"
            else self.random.randint(90, 130)
        )
        self.targets.append(
            Target(
                target_type=target_type,
                center=self.random_position(size),
                size=size,
                expires_at_ms=pygame.time.get_ticks() + TARGET_LIFETIME_MS,
            )
        )

    def random_position(self, size: int) -> tuple[int, int]:
        margin = size // 2 + 20
        x = self.random.randint(margin, WINDOW_WIDTH - margin)
        y = self.random.randint(
            HEADER_HEIGHT + margin,
            WINDOW_HEIGHT - margin,
        )
        return x, y

    def expire_targets(self) -> None:
        current_ms = pygame.time.get_ticks()
        self.targets = [
            target for target in self.targets if current_ms < target.expires_at_ms
        ]

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        pygame.draw.rect(
            self.screen,
            HEADER_BACKGROUND,
            (0, 0, WINDOW_WIDTH, HEADER_HEIGHT),
        )
        for target in self.targets:
            target.draw(self.screen)
        self.draw_status_bar()
        pygame.display.flip()

    def draw_status_bar(self) -> None:
        remaining_s = max(0, (self.ends_at_ms - pygame.time.get_ticks()) / 1000)
        instruction_surface = self.title_font.render(
            self.current_instruction,
            True,
            YELLOW,
        )
        score_surface = self.normal_font.render(f"得分：{self.score}", True, WHITE)
        time_surface = self.normal_font.render(
            f"时间：{remaining_s:05.1f}",
            True,
            WHITE,
        )
        self.screen.blit(
            instruction_surface,
            (
                WINDOW_WIDTH // 2 - instruction_surface.get_width() // 2,
                45,
            ),
        )
        self.screen.blit(score_surface, (20, 15))
        self.screen.blit(
            time_surface,
            (WINDOW_WIDTH - time_surface.get_width() - 20, 15),
        )

    def show_final_screen(self) -> None:
        final_started = time.monotonic()
        while self.running and time.monotonic() - final_started < 5:
            self.handle_events()
            self.screen.fill(BACKGROUND)
            self.draw_center_text("本轮结束", WHITE, WINDOW_HEIGHT // 2 - 40)
            self.draw_center_text(
                f"最终得分：{self.score}",
                YELLOW,
                WINDOW_HEIGHT // 2 + 20,
            )
            pygame.display.flip()
            self.clock.tick(FPS)

    def draw_center_text(
        self,
        text: str,
        color: tuple[int, int, int],
        center_y: int,
    ) -> None:
        surface = self.title_font.render(text, True, color)
        rect = surface.get_rect(center=(WINDOW_WIDTH // 2, center_y))
        self.screen.blit(surface, rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动红蓝目标点击实验。")
    parser.add_argument("--stage", required=True, choices=STAGES.keys())
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RedBlueGame(stage_name=args.stage, seed=args.seed).start()


if __name__ == "__main__":
    main()
