"""Dock 训练机床 — 接收训练订单，完成 TNN 训练全流程。"""
from __future__ import annotations

import json
import random
import time
from typing import Any

import torch
import torch.nn as nn

from eve.core.tnn_base import TNNBase, TNNDescriptor, SourceRef
from eve.dock.order import TrainingOrder, TrainingResult
from eve.dock.tiny_nn import create_tnn_from_order


class Trainer:
    """Dock 训练机床。

    职责：
    - 接收训练订单
    - 从 Memory 获取训练数据
    - 调用 teacher 产生标签
    - 执行 TNN 训练
    - 离线评估
    - 保存 TNN 到 TNN Store
    - 生成训练报告

    不负责：
    - 决定训练什么（LLM/用户的事）
    - 运行时加载 TNN（Core 的事）
    - 修改 TNNGraph（Core 的事）
    """

    def __init__(self, tnn_store, memorizer):
        self._tnn_store = tnn_store
        self._memorizer = memorizer
        self._queue: list[TrainingOrder] = []
        self._current_order: TrainingOrder | None = None
        self._running = False
        self._results: list[TrainingResult] = []

    # ── 队列管理 ─────────────────────────────────────────

    def enqueue(self, order: TrainingOrder) -> None:
        """将训练订单加入队列。"""
        order.status = "pending"
        self._queue.append(order)
        # 按优先级排序: high > medium > low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self._queue.sort(key=lambda o: priority_order.get(o.priority, 1))

    def has_pending(self) -> bool:
        """是否还有待处理的订单。"""
        return len(self._queue) > 0

    def get_queue_size(self) -> int:
        """返回队列中的订单数。"""
        return len(self._queue)

    @property
    def is_training(self) -> bool:
        """当前是否正在训练。"""
        return self._running

    @property
    def current_order(self) -> TrainingOrder | None:
        """当前正在处理的订单。"""
        return self._current_order

    @property
    def results(self) -> list[TrainingResult]:
        """所有已完成（成功或失败）的训练结果。"""
        return self._results

    def stats(self) -> dict:
        """返回训练机床统计信息。"""
        success_count = sum(1 for r in self._results if r.success)
        fail_count = sum(1 for r in self._results if not r.success)
        return {
            "queue_size": len(self._queue),
            "is_training": self._running,
            "current_order": self._current_order.order_id if self._current_order else None,
            "total_completed": len(self._results),
            "success_count": success_count,
            "fail_count": fail_count,
        }

    # ── 训练主流程 ───────────────────────────────────────

    def process_one(self, model_adapters: dict | None = None) -> TrainingResult:
        """处理一个训练订单。

        流程：
        1. 从队列取出订单
        2. 从 Memory 获取训练数据
        3. 调用 teacher 产生标签
        4. 创建/加载 TNN
        5. 训练循环
        6. 离线评估
        7. 保存到 TNN Store
        8. 生成训练报告 MemoryUnit
        9. 返回结果
        """
        if not self._queue:
            return TrainingResult(
                order_id="",
                success=False,
                error_message="No orders in queue",
            )

        self._running = True
        self._current_order = self._queue.pop(0)
        order = self._current_order
        order.status = "running"
        t_start = time.monotonic_ns()

        try:
            # 2. 从 Memory 获取训练数据
            inputs_list, raw_data = self._fetch_training_data(order)

            if not inputs_list:
                order.status = "failed"
                result = TrainingResult(
                    order_id=order.order_id,
                    success=False,
                    tnn_id=order.target_tnn_id,
                    error_message="No training data retrieved from memory",
                )
                self._results.append(result)
                self._running = False
                return result

            # 3. 调用 teacher 产生标签
            labels = self._call_teacher(order, raw_data, model_adapters)

            # 4. 创建/加载 TNN
            input_dim = self._estimate_input_dim(inputs_list[0])
            tnn = create_tnn_from_order(order, input_dim_estimate=input_dim)

            # 如果不是新建，尝试从 store 加载已有版本
            if not order.create_new:
                existing = self._tnn_store.load_tnn(order.target_tnn_id)
                if existing is not None:
                    tnn = existing

            # 5. 训练循环
            if order.structure_hint in ("cnn", "conv_small"):
                train_result = self._train_cnn_tnn(order, tnn, inputs_list, labels)
            else:
                train_result = self._train_mlp_tnn(order, tnn, inputs_list, labels)

            if not train_result.success:
                order.status = "failed"
                self._results.append(train_result)
                self._running = False
                return train_result

            # 6. 离线评估
            eval_metrics = self._evaluate(tnn, inputs_list, labels)
            train_result.eval_metrics = eval_metrics

            # 7. 保存到 TNN Store
            version = order.created_at_ns % 1_000_000_000  # naive version from timestamp
            saved_dir = self._tnn_store.save_new_version(
                order.target_tnn_id, tnn, version
            )
            train_result.version = version

            # 8. 生成训练报告 MemoryUnit
            report_id = self._generate_report(order, train_result, saved_dir)
            if report_id:
                train_result.memory_reports.append(report_id)

            elapsed_ns = time.monotonic_ns() - t_start
            train_result.latency_ms = elapsed_ns / 1_000_000.0

            order.status = "completed"
            self._results.append(train_result)
            self._running = False
            return train_result

        except Exception as e:
            order.status = "failed"
            elapsed_ns = time.monotonic_ns() - t_start
            result = TrainingResult(
                order_id=order.order_id,
                success=False,
                tnn_id=order.target_tnn_id,
                error_message=str(e),
                latency_ms=elapsed_ns / 1_000_000.0,
            )
            self._results.append(result)
            self._running = False
            return result

    # ── 数据获取 ─────────────────────────────────────────

    def _fetch_training_data(self, order: TrainingOrder) -> tuple[list[torch.Tensor], list[Any]]:
        """从 Memory 获取训练数据，返回 (tensor_list, raw_data_list)。"""
        inputs_list: list[torch.Tensor] = []
        raw_data: list[Any] = []

        for mem_id in order.training_data:
            data = self._memorizer.read(mem_id)
            if data is None:
                continue
            raw_data.append(data)

            tensor = self._data_to_tensor(data)
            if tensor is not None:
                inputs_list.append(tensor)

        return inputs_list, raw_data

    @staticmethod
    def _data_to_tensor(data: Any) -> torch.Tensor | None:
        """将任意 Memory payload 转为固定维度的 tensor。"""
        try:
            if isinstance(data, torch.Tensor):
                return data.float()
            elif isinstance(data, (int, float)):
                return torch.tensor([float(data)], dtype=torch.float32)
            elif isinstance(data, str):
                # 简单哈希 → 64 维向量
                vec = torch.zeros(64, dtype=torch.float32)
                for i, ch in enumerate(data):
                    vec[i % 64] += float(ord(ch) % 100) / 100.0
                return vec
            elif isinstance(data, (list, tuple)):
                flat = []
                for item in data:
                    if isinstance(item, (int, float)):
                        flat.append(float(item))
                if flat:
                    t = torch.tensor(flat, dtype=torch.float32)
                    # pad or truncate to 64
                    if t.shape[0] < 64:
                        t = torch.nn.functional.pad(t, (0, 64 - t.shape[0]))
                    elif t.shape[0] > 64:
                        t = t[:64]
                    return t
                return None
            elif isinstance(data, dict):
                # 序列化为 JSON 再编码
                return Trainer._data_to_tensor(json.dumps(data, ensure_ascii=False))
            else:
                import numpy as np
                arr = np.asarray(data, dtype=np.float32).flatten()
                t = torch.from_numpy(arr)
                if t.shape[0] < 64:
                    t = torch.nn.functional.pad(t, (0, 64 - t.shape[0]))
                elif t.shape[0] > 64:
                    t = t[:64]
                return t
        except Exception:
            return None

    @staticmethod
    def _estimate_input_dim(tensor: torch.Tensor) -> int:
        """估计输入维度。"""
        return int(tensor.reshape(-1).shape[0])

    # ── Teacher ──────────────────────────────────────────

    def _call_teacher(
        self, order: TrainingOrder, data: list[Any], model_adapters: dict | None = None
    ) -> list[torch.Tensor]:
        """调用 teacher 产生标签。

        teacher 类型处理：
        - "rule" → 基于规则的简单标签生成
        - "local_llm" → 使用本地 LLM 适配器
        - 其他 → 回退到 rule
        """
        if order.teacher == "rule":
            return self._rule_teacher(data, order)
        elif order.teacher == "local_llm" and model_adapters:
            llm = model_adapters.get("local_llm")
            if llm is not None:
                return self._llm_teacher(llm, data, order)
        # 默认回退到 rule
        return self._rule_teacher(data, order)

    def _rule_teacher(self, data: list[Any], order: TrainingOrder) -> list[torch.Tensor]:
        """基于规则的标签生成器。
        
        为每个数据样本生成一个标签 tensor：
        - 文本: 统计长度、行数等特征作为标签
        - 数值: 统计均值、方差等
        - 其他: 随机标签
        """
        labels: list[torch.Tensor] = []
        for item in data:
            if isinstance(item, str):
                features = torch.tensor([
                    float(len(item)) / 1000.0,
                    float(item.count("\n")) / 100.0,
                    float(len(item.split())) / 200.0,
                    float(sum(1 for c in item if c.isdigit())) / max(1, len(item)),
                ], dtype=torch.float32)
                # pad to 16
                if features.shape[0] < 16:
                    features = torch.nn.functional.pad(features, (0, 16 - features.shape[0]))
                labels.append(features)
            elif isinstance(item, (int, float)):
                labels.append(torch.tensor([float(item), float(item) * 0.5, 0.0, 0.0], dtype=torch.float32))
            elif isinstance(item, (list, tuple)):
                vals = [float(x) for x in item if isinstance(x, (int, float))]
                if vals:
                    t = torch.tensor(vals[:16], dtype=torch.float32)
                    if t.shape[0] < 16:
                        t = torch.nn.functional.pad(t, (0, 16 - t.shape[0]))
                    labels.append(t)
                else:
                    labels.append(torch.zeros(16, dtype=torch.float32))
            elif isinstance(item, dict):
                labels.append(torch.tensor([
                    float(len(item)),
                    float(len(json.dumps(item, ensure_ascii=False))),
                    0.5, 0.5,
                ], dtype=torch.float32))
            else:
                labels.append(torch.rand(16, dtype=torch.float32))
        return labels

    def _llm_teacher(
        self, llm, data: list[Any], order: TrainingOrder
    ) -> list[torch.Tensor]:
        """使用 LLM 适配器产生标签。
        
        llm 预期有 generate(prompt) → str 接口。
        将自然语言标签转换为 tensor。
        """
        labels: list[torch.Tensor] = []
        for item in data:
            prompt = order.teacher_prompt or "Label this data with a numeric score (0-1): "
            item_str = str(item)[:2000]
            full_prompt = f"{prompt}\n\nData: {item_str}\n\nScore:"
            try:
                llm_output = llm.generate(full_prompt)
                # 尝试从输出提取数值
                score = self._extract_score(llm_output)
                labels.append(torch.tensor([score], dtype=torch.float32))
            except Exception:
                # LLM 调用失败，回退到 rule
                return self._rule_teacher(data, order)
        return labels

    @staticmethod
    def _extract_score(text: str) -> float:
        """从 LLM 输出中提取第一个 0-1 之间的数值。"""
        import re
        numbers = re.findall(r"([0-9]*\.?[0-9]+)", text)
        for n in numbers:
            try:
                val = float(n)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
        # 尝试从 0-100 范围映射
        for n in numbers:
            try:
                val = float(n)
                if 0.0 <= val <= 100.0:
                    return val / 100.0
            except ValueError:
                continue
        return 0.5  # 默认值

    # ── 训练方法 ─────────────────────────────────────────

    def _train_mlp_tnn(
        self, order: TrainingOrder, tnn: TNNBase,
        inputs: list[torch.Tensor], labels: list[torch.Tensor],
    ) -> TrainingResult:
        """训练 MLP 类型 TNN。"""
        if not inputs or not labels:
            return TrainingResult(
                order_id=order.order_id,
                success=False,
                tnn_id=order.target_tnn_id,
                error_message="Empty inputs or labels",
            )

        # 确保 inputs 和 labels 对齐
        n = min(len(inputs), len(labels))
        inputs = inputs[:n]
        labels = labels[:n]

        if n == 0:
            return TrainingResult(
                order_id=order.order_id,
                success=False,
                tnn_id=order.target_tnn_id,
                error_message="No aligned samples",
            )

        tnn.train()
        optimizer = torch.optim.Adam(tnn.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        epochs = min(50, max(10, n * 2))
        batch_size = min(16, n)

        for epoch in range(epochs):
            total_loss = 0.0
            indices = list(range(n))
            random.shuffle(indices)

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_inputs = [inputs[i] for i in batch_idx]
                batch_labels = [labels[i] for i in batch_idx]

                x_batch = torch.stack([x.flatten() for x in batch_inputs])
                y_batch = torch.stack([y.flatten() for y in batch_labels])

                # 确保维度匹配
                in_dim = getattr(tnn, "_input_dim", 64)
                if x_batch.shape[1] < in_dim:
                    x_batch = torch.nn.functional.pad(x_batch, (0, in_dim - x_batch.shape[1]))
                elif x_batch.shape[1] > in_dim:
                    x_batch = x_batch[:, :in_dim]

                out_dim = getattr(tnn, "_output_dim", 16)
                if y_batch.shape[1] < out_dim:
                    y_batch = torch.nn.functional.pad(y_batch, (0, out_dim - y_batch.shape[1]))
                elif y_batch.shape[1] > out_dim:
                    y_batch = y_batch[:, :out_dim]

                optimizer.zero_grad()
                # 构造 dict 输入
                output_names = getattr(tnn, "_output_names", None) or ["output"]
                output_name = output_names[0] if output_names else "output"
                predictions = tnn({output_name: x_batch})[output_name]
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, ((n + batch_size - 1) // batch_size))

        tnn.eval()
        return TrainingResult(
            order_id=order.order_id,
            success=True,
            tnn_id=order.target_tnn_id,
            eval_metrics={"final_loss": round(avg_loss, 6), "epochs": epochs, "samples": n},
        )

    def _train_cnn_tnn(
        self, order: TrainingOrder, tnn: TNNBase,
        inputs: list[torch.Tensor], labels: list[torch.Tensor],
    ) -> TrainingResult:
        """训练 CNN 类型 TNN。"""
        if not inputs or not labels:
            return TrainingResult(
                order_id=order.order_id,
                success=False,
                tnn_id=order.target_tnn_id,
                error_message="Empty inputs or labels",
            )

        n = min(len(inputs), len(labels))
        inputs = inputs[:n]
        labels = labels[:n]

        if n == 0:
            return TrainingResult(
                order_id=order.order_id,
                success=False,
                tnn_id=order.target_tnn_id,
                error_message="No aligned samples",
            )

        tnn.train()
        optimizer = torch.optim.Adam(tnn.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        epochs = min(50, max(10, n * 2))
        batch_size = min(8, n)

        for epoch in range(epochs):
            total_loss = 0.0
            indices = list(range(n))
            random.shuffle(indices)

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_inputs = [inputs[i] for i in batch_idx]
                batch_labels = [labels[i] for i in batch_idx]

                # 将 inputs reshape 为 (B, C, H, W) 格式
                x_list = []
                for xi in batch_inputs:
                    flat = xi.flatten()
                    side = int(flat.shape[0] ** 0.5)
                    if side * side == flat.shape[0] and side >= 4:
                        x_reshaped = flat[:side * side].reshape(1, side, side)
                    else:
                        # pad 到最近平方数
                        target = max(4, int(flat.shape[0] ** 0.5) + 1)
                        padded = torch.zeros(target * target)
                        padded[:flat.shape[0]] = flat[:target * target]
                        x_reshaped = padded.reshape(1, target, target)
                    x_list.append(x_reshaped)

                x_batch = torch.stack(x_list)
                y_batch = torch.stack([y.flatten() for y in batch_labels])

                # 确保输出维度匹配
                out_dim = getattr(tnn, "_output_dim", 64)
                if y_batch.shape[1] < out_dim:
                    y_batch = torch.nn.functional.pad(y_batch, (0, out_dim - y_batch.shape[1]))
                elif y_batch.shape[1] > out_dim:
                    y_batch = y_batch[:, :out_dim]

                optimizer.zero_grad()
                output_names = getattr(tnn, "_output_names", None) or ["output"]
                output_name = output_names[0] if output_names else "output"
                predictions = tnn({output_name: x_batch})[output_name]
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, ((n + batch_size - 1) // batch_size))

        tnn.eval()
        return TrainingResult(
            order_id=order.order_id,
            success=True,
            tnn_id=order.target_tnn_id,
            eval_metrics={"final_loss": round(avg_loss, 6), "epochs": epochs, "samples": n},
        )

    # ── 评估 ─────────────────────────────────────────────

    def _evaluate(
        self, tnn: TNNBase, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> dict[str, float]:
        """简单离线评估：计算 MSE loss。"""
        tnn.eval()
        with torch.no_grad():
            n = min(len(inputs), len(labels))
            if n == 0:
                return {"eval_loss": 0.0, "eval_samples": 0}

            total_loss = 0.0
            output_names = getattr(tnn, "_output_names", None) or ["output"]
            output_name = output_names[0] if output_names else "output"

            for i in range(n):
                x = inputs[i].flatten().unsqueeze(0)
                y = labels[i].flatten().unsqueeze(0)
                in_dim = getattr(tnn, "_input_dim", 64)
                if x.shape[1] < in_dim:
                    x = torch.nn.functional.pad(x, (0, in_dim - x.shape[1]))
                elif x.shape[1] > in_dim:
                    x = x[:, :in_dim]

                # 对于 CNN，需要额外 reshape
                if isinstance(tnn, type(tnn)) and hasattr(tnn, "conv"):
                    side = int(x.shape[1] ** 0.5)
                    if side * side == x.shape[1] and side >= 4:
                        x = x.reshape(1, 1, side, side)
                    else:
                        target = max(4, int(x.shape[1] ** 0.5) + 1)
                        padded = torch.zeros(1, target * target)
                        padded[0, :x.shape[1]] = x[0, :target * target]
                        x = padded.reshape(1, 1, target, target)

                out_dim = getattr(tnn, "_output_dim", 16)
                if y.shape[1] < out_dim:
                    y = torch.nn.functional.pad(y, (0, out_dim - y.shape[1]))
                elif y.shape[1] > out_dim:
                    y = y[:, :out_dim]

                pred = tnn({output_name: x})[output_name]
                loss = nn.functional.mse_loss(pred, y)
                total_loss += loss.item()

            return {"eval_loss": round(total_loss / n, 6), "eval_samples": n}

    # ── 报告生成 ─────────────────────────────────────────

    def _generate_report(
        self, order: TrainingOrder, result: TrainingResult, saved_dir: str
    ) -> str:
        """生成训练报告并存入 Memory，返回 MemoryID。"""
        report = {
            "order_id": order.order_id,
            "tnn_id": result.tnn_id,
            "version": result.version,
            "purpose": order.purpose,
            "teacher": order.teacher,
            "structure_hint": order.structure_hint,
            "eval_metrics": result.eval_metrics,
            "latency_ms": result.latency_ms,
            "saved_dir": saved_dir,
            "training_samples": len(order.training_data),
            "success": result.success,
            "error": result.error_message,
        }
        try:
            report_id = self._memorizer.create(report, payload_type="json")
            return report_id
        except Exception:
            return ""
