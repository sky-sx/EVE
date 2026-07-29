"""Interactive red-circle / blue-triangle growth environment for EVE."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eve.main import EVEApplication, EVEControlWindow


INSTRUCTIONS = {
    "red_only": "点击红色圆形",
    "red_and_blue": "点击红色圆形和蓝色三角形",
    "instruction_driven": "保持红色圆形能力，同时点击蓝色三角形",
}


class ShapeCanvas(QWidget):
    def __init__(
        self,
        application: Callable[[], EVEApplication],
        *,
        mode: str,
        event_log: Path,
        seed: int,
    ) -> None:
        super().__init__()
        self._application = application
        self.mode = mode
        self.event_log = event_log
        self.random = random.Random(seed)
        self.targets: list[dict[str, Any]] = []
        self.score = 0
        self.status_callback: Callable[[str], None] = lambda _text: None
        self.setMinimumSize(760, 620)
        self.setStyleSheet("background: white;")
        self._spawn_all()

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.mode]

    def _spawn_all(self) -> None:
        kinds = ["red_circle"]
        if self.mode in {"red_and_blue", "instruction_driven"}:
            kinds.append("blue_triangle")
        self.targets = [self._new_target(kind) for kind in kinds]
        self.update()

    def _new_target(self, kind: str) -> dict[str, Any]:
        size = self.random.randint(70, 120)
        width = max(self.width(), 760)
        height = max(self.height(), 620)
        x = self.random.randint(30, max(31, width - size - 30))
        y = self.random.randint(30, max(31, height - size - 30))
        target = {
            "target_id": f"{kind}_{uuid.uuid4().hex[:10]}",
            "class": kind,
            "rect": QRectF(float(x), float(y), float(size), float(size)),
            "spawned_at_ns": time.monotonic_ns(),
        }
        self._log(
            {
                "event": "target_spawned",
                "target_id": target["target_id"],
                "class": kind,
                "timestamp_ns": target["spawned_at_ns"],
            }
        )
        return target

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for target in self.targets:
            rect = target["rect"]
            if target["class"] == "red_circle":
                painter.setPen(QPen(QColor("#9b111e"), 3))
                painter.setBrush(QColor("#ef233c"))
                painter.drawEllipse(rect)
            else:
                painter.setPen(QPen(QColor("#003f88"), 3))
                painter.setBrush(QColor("#168aad"))
                painter.drawPolygon(self._triangle(rect))
        painter.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = event.position()
        hit_target = next(
            (target for target in self.targets if self._contains(target, point)),
            None,
        )
        if hit_target is not None:
            self.score += 1
        feedback = {
            "task_id": (
                "red_circle_click"
                if self.mode == "red_only"
                else "red_blue_shapes_click"
            ),
            "instruction": self.instruction,
            "target_classes": (
                [hit_target["class"]] if hit_target is not None else []
            ),
            "target_id": (
                hit_target["target_id"] if hit_target is not None else None
            ),
            "hit": hit_target is not None,
            "score_delta": 1 if hit_target is not None else 0,
            "score_total": self.score,
            "reward": 1.0 if hit_target is not None else -1.0,
            "x": int(point.x()),
            "y": int(point.y()),
            "timestamp_ns": time.monotonic_ns(),
        }
        source, memory_id = self._publish_feedback(feedback)
        self._log(
            {
                "event": "click_result",
                **feedback,
                "source": source,
                "experience_memory_id": memory_id,
            }
        )
        self.status_callback(
            f"score={self.score} | hit={feedback['hit']} | "
            f"source={source} | experience={memory_id or '-'}"
        )
        if hit_target is not None:
            index = self.targets.index(hit_target)
            self.targets[index] = self._new_target(hit_target["class"])
            self.update()

    def _publish_feedback(
        self, feedback: dict[str, Any]
    ) -> tuple[str, str | None]:
        application = self._application()
        pending = application.state.get("pending_experiences", {})
        try:
            if pending:
                candidate_id = next(reversed(pending))
                return (
                    "eve",
                    application.core.submit_environment_feedback(
                        candidate_id, feedback
                    ),
                )
            teacher = application.state.get("last_teacher_visual_result")
            if (
                isinstance(teacher, dict)
                and teacher.get("label_status") == "valid"
            ):
                return (
                    "human_demonstration",
                    application.core.record_teacher_demonstration(feedback),
                )
        except Exception as exc:
            self.status_callback(f"feedback error: {type(exc).__name__}: {exc}")
        return "unbound", None

    @staticmethod
    def _triangle(rect: QRectF) -> QPolygonF:
        return QPolygonF(
            [
                QPointF(rect.center().x(), rect.top()),
                QPointF(rect.right(), rect.bottom()),
                QPointF(rect.left(), rect.bottom()),
            ]
        )

    def _contains(self, target: dict[str, Any], point: QPointF) -> bool:
        rect = target["rect"]
        if target["class"] == "red_circle":
            radius_x = rect.width() / 2
            radius_y = rect.height() / 2
            dx = (point.x() - rect.center().x()) / radius_x
            dy = (point.y() - rect.center().y()) / radius_y
            return dx * dx + dy * dy <= 1.0
        path = QPainterPath()
        polygon = self._triangle(rect)
        path.addPolygon(polygon)
        path.closeSubpath()
        return path.contains(point)

    def _log(self, record: dict[str, Any]) -> None:
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ExperimentWindow(QMainWindow):
    def __init__(
        self,
        control: Any,
        *,
        mode: str,
        event_log: Path,
        seed: int,
    ) -> None:
        super().__init__()
        self.control = control
        self.setWindowTitle("EVE Red / Blue Shape Growth Experiment")
        root = QWidget()
        layout = QVBoxLayout(root)
        self.instruction = QLabel(INSTRUCTIONS[mode])
        self.instruction.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.instruction)
        self.status = QLabel("等待冷启动、教师标签或点击。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.canvas = ShapeCanvas(
            lambda: self.control.application,
            mode=mode,
            event_log=event_log,
            seed=seed,
        )
        self.canvas.status_callback = self.status.setText
        layout.addWidget(self.canvas)
        buttons = QHBoxLayout()
        review = QPushButton("请求VLM教师标签")
        review.clicked.connect(self._request_teacher)
        train_red = QPushButton("训练红圆TNN")
        train_red.clicked.connect(
            lambda: self._train(
                "red_circle",
                "red_circle_click",
                "red_circle_locator",
            )
        )
        train_blue = QPushButton("训练蓝三角TNN")
        train_blue.setEnabled(mode != "red_only")
        train_blue.clicked.connect(
            lambda: self._train(
                "blue_triangle",
                "red_blue_shapes_click",
                "blue_triangle_locator",
            )
        )
        train_qnn = QPushButton("训练动作评价QNN")
        train_qnn.clicked.connect(self._train_qnn)
        reset = QPushButton("重置目标")
        reset.clicked.connect(self.canvas._spawn_all)
        for button in (review, train_red, train_blue, train_qnn, reset):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.setCentralWidget(root)
        self.resize(820, 760)

    def _request_teacher(self) -> None:
        try:
            request_id = self.control.application.core.submit_teacher_review(
                prompt=(
                    "识别屏幕中所有红色圆形和蓝色三角形。"
                    "输出objects数组，每项包含class、bbox、center、confidence。"
                )
            )
            self.status.setText(f"VLM request: {request_id}")
        except Exception as exc:
            self.status.setText(f"VLM error: {type(exc).__name__}: {exc}")

    def _train(
        self,
        target_class: str,
        task_id: str,
        tnn_id: str,
    ) -> None:
        try:
            order_id = self.control.application.core.request_shape_training(
                task_id=task_id,
                target_class=target_class,
                target_tnn_id=tnn_id,
            )
            self.status.setText(f"TrainingOrder: {order_id}")
        except Exception as exc:
            self.status.setText(f"Dock error: {type(exc).__name__}: {exc}")

    def _train_qnn(self) -> None:
        task_id = (
            "red_circle_click"
            if self.canvas.mode == "red_only"
            else "red_blue_shapes_click"
        )
        try:
            order_id = self.control.application.core.request_qnn_training(
                task_id=task_id,
            )
            self.status.setText(f"QNN TrainingOrder: {order_id}")
        except Exception as exc:
            self.status.setText(f"QNN Dock error: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=tuple(INSTRUCTIONS),
        default="red_only",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", default="runs/red_blue_experiment")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    qt_application = QApplication.instance() or QApplication(sys.argv[:1])

    def factory() -> EVEApplication:
        return EVEApplication(
            profile="control",
            mode="real",
            run_dir=run_dir,
            allow_mock_actions=False,
        )

    control = EVEControlWindow.create(factory(), application_factory=factory)
    experiment = ExperimentWindow(
        control,
        mode=args.mode,
        event_log=run_dir / "environment.jsonl",
        seed=args.seed,
    )
    control.show()
    experiment.show()
    control.move(20, 20)
    experiment.move(1420, 20)
    return int(qt_application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
