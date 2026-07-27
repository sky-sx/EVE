"""EVE 中所有 Tiny Neural Network 的最小统一接口。

tinynn.py 只定义单一基类 TinyNN，每一个具体 TNN 都必须继承它。
它规定 TNN 必须能做什么（声明输入输出、执行前向推理、完成一次训练步骤、保存/加载权重），
但不规定内部结构如何实现。

不强制的架构（MLP、CNN、GRU、残差、门控、离散……），
每个具体 TNN 在自己的 ``model.py`` 中实现自身参数、forward、backward 和 training_step。
"""

from __future__ import annotations

import copy
from typing import Any

import torch


class TinyNN(torch.nn.Module):
    """所有可训练 Tiny Neural Network 的最小统一接口。

    TinyNN 本身不是可直接使用的网络。具体 TNN 继承它并自行提供：
    网络层、``forward()``、``training_step()`` 以及所需的任何自定义反向传播逻辑。

    完整生命周期
    -------------
    1. LLM 决定需要形成一个新 TNN
    2. LLM 生成具体 TNN 的 ``model.py``
    3. 具体 class 继承 ``TinyNN``
    4. ``trainer.py`` 创建实例并调用 ``training_step()``
    5. 训练通过后，``save_weights()`` 保存权重
    6. ``Core`` 加载 ``model.py`` 创建 TNN 实例，调用 ``load_weights()`` 加载权重，运行时调用 ``infer()``
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        tnn_id: str,
        version: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """保存身份元数据。具体网络层由子类创建。

        参数
        ----------
        tnn_id:
            唯一身份名称，例如 ``"mouse_reflex_0001"``。
        version:
            实现版本字符串。同一个 *tnn_id* 可以有不同的版本，
            对应不同的网络结构和权重。
        input_schema:
            声明 TNN 接受什么输入，例如::

                {
                    "raw_screen": {"dtype": "uint8", "shape": [1080, 1920, 4]},
                    "cursor_point": {"dtype": "int32", "shape": [2]},
                }
        output_schema:
            声明 TNN 输出什么，例如::

                {
                    "mouse_delta": {"dtype": "float32", "shape": [2]},
                    "click_tendency": {"dtype": "float32", "shape": [1]},
                }
        """
        super().__init__()
        self._tnn_id = tnn_id
        self._version = version
        self._input_schema = copy.deepcopy(input_schema)
        self._output_schema = copy.deepcopy(output_schema)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def tnn_id(self) -> str:
        """该 TNN 的身份名称。"""
        return self._tnn_id

    @property
    def version(self) -> str:
        """当前实现版本。"""
        return self._version

    # ------------------------------------------------------------------
    # Schema 查询
    # ------------------------------------------------------------------

    def get_input_schema(self) -> dict[str, Any]:
        """返回 input_schema 的独立副本。

        供 ``trainer.py`` 检查训练数据和 ``Core`` 确认上下游接口使用。
        返回深拷贝，避免外部代码修改返回值时改变 TNN 内部定义。
        """
        return copy.deepcopy(self._input_schema)

    def get_output_schema(self) -> dict[str, Any]:
        """返回 output_schema 的独立副本。"""
        return copy.deepcopy(self._output_schema)

    # ------------------------------------------------------------------
    # 前向传播（子类必须覆写）
    # ------------------------------------------------------------------

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """执行该 TNN 的前向计算。

        每一个具体 TNN **必须**覆写此方法。基类实现直接抛出 ``NotImplementedError``。

        参数
        ----------
        inputs:
            符合该 TNN ``input_schema`` 的字典，由调用方（``trainer.py`` 或 ``Core``）组装。

        返回
        -------
        dict
            符合 ``output_schema`` 的字典。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 forward()"
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """运行期统一推理入口，在 eval 模式下关闭梯度记录执行推理。

        这是 ``Core`` 使用的统一运行时入口::

            Core 提供输入
              → TinyNN.infer(inputs)
                → eval 模式
                → torch.no_grad()
                → 具体 forward(inputs)
              → 返回输出
            Core 将输出写入当前运行期节点关系

        ``infer()`` 不改变权重、不调用下游 TNN、也不执行动作。
        """
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

    # ------------------------------------------------------------------
    # 训练步骤（子类必须覆写）
    # ------------------------------------------------------------------

    def training_step(self, batch: Any) -> dict[str, Any]:
        """执行该 TNN 的一次训练步骤。

        每一个具体 TNN **必须**覆写此方法。``batch`` 由 ``trainer.py`` 根据当前训练订单准备。

        该方法负责完成该具体 TNN 的完整训练计算：
        读取 batch、执行 forward、计算 loss、执行 backward、更新参数。
        返回结果至少包含 ``"loss"`` 键::

            {"loss": float}

        也可以包含额外指标::

            {"loss": float, "accuracy": float}
            {"loss": float, "position_error": float, "overshoot": float}

        基类不规定统一的 batch 格式、损失函数或 optimizer，
        因此分类、回归、控制、预测、自编码、离散 TNN 都可以使用同一个最小训练入口。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 training_step()"
        )

    # ------------------------------------------------------------------
    # 权重持久化
    # ------------------------------------------------------------------

    def evaluation_step(self, batch: Any) -> dict[str, Any]:
        """Evaluate one batch using semantics supplied by the concrete TNN."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluation_step()"
        )

    def save_weights(self, path: str) -> None:
        """将 TNN 的 ``state_dict()`` 保存到指定路径 *path*。

        仅保存网络参数、可训练权重和持久 buffer。
        训练订单、报告、运行期节点关系、MemoryID、运行指标和 TNN 描述文件
        由 ``trainer.py`` 和 ``Memory`` 负责。
        """
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: Any = None) -> None:
        """从 *path* 加载权重到当前 TNN 实例。

        调用前，具体 TNN 的代码和网络结构**必须**已经创建。
        如果权重文件结构与当前模型不匹配，加载将直接失败报错
        （PyTorch 的 ``load_state_dict`` 默认使用 ``strict=True``）。

        参数
        ----------
        path:
            指向 ``.pt`` 权重文件的文件系统路径。
        map_location:
            透传给 ``torch.load``，用于设备重映射。
        """
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
