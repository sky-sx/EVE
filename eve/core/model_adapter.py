"""模型适配层 — 统一接口管理本地 LLM、VLM 和 YOLO。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import time
import json
import re
import gc


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class ModelStatus:
    available: bool
    loaded: bool = False
    device: str = "cpu"
    vram_usage_mb: float = 0.0
    last_error: str = ""
    last_inference_ms: float = 0.0


@dataclass
class InferenceResult:
    success: bool
    output: str              # raw text output
    structured: dict | None  # parsed JSON if applicable
    inference_time_ms: float
    error: str = ""
    tokens_used: int = 0


@dataclass
class ObjectDetection:
    """YOLO 单次检测结果。"""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int = -1


@dataclass
class DetectionResult:
    success: bool
    detections: list[ObjectDetection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    error: str = ""
    image_size: tuple[int, int] = (0, 0)


# ── JSON 解析工具 ─────────────────────────────────────────

def _parse_json_from_text(text: str) -> dict | None:
    """从 LLM/VLM 输出文本中提取 JSON 对象。"""
    # 先尝试匹配 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 再尝试匹配首尾花括号
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_llm_output(output: dict, schema: dict) -> tuple[bool, str]:
    """校验 LLM 输出是否符合预期 schema。返回 (valid, error)。"""
    for key, expected_type in schema.items():
        if key not in output:
            return False, f"missing key: {key}"
        if expected_type == "str" and not isinstance(output[key], str):
            return False, f"key {key} should be str, got {type(output[key]).__name__}"
        if expected_type == "list" and not isinstance(output[key], list):
            return False, f"key {key} should be list, got {type(output[key]).__name__}"
        if expected_type == "bool" and not isinstance(output[key], bool):
            return False, f"key {key} should be bool, got {type(output[key]).__name__}"
        if expected_type == "dict" and not isinstance(output[key], dict):
            return False, f"key {key} should be dict, got {type(output[key]).__name__}"
        if expected_type == "number" and not isinstance(output[key], (int, float)):
            return False, f"key {key} should be number, got {type(output[key]).__name__}"
    return True, ""


# ── 基类 ──────────────────────────────────────────────────

class BaseModelAdapter(ABC):
    """所有模型适配器的基类。

    子类必须实现 detect / load / unload / infer。
    """

    def __init__(self, name: str):
        self.name = name
        self.status = ModelStatus(available=False)

    @abstractmethod
    def detect(self) -> bool:
        """检测模型是否可用。返回 True/False。"""
        ...

    @abstractmethod
    def load(self, **kwargs) -> bool:
        """加载模型到内存/显存。返回 True/False。"""
        ...

    @abstractmethod
    def unload(self) -> None:
        """卸载模型释放资源。"""
        ...

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """执行推理。"""
        ...


# ── LLM 适配器 ────────────────────────────────────────────

class LLMAdapter(BaseModelAdapter):
    """本地 LLM 适配器。

    支持 DeepSeek、Qwen 等。自动检测模型路径，处理量化加载。
    如果模型不可用，返回明确的错误信息。

    Usage:
        adapter = LLMAdapter(model_path="eve/core/deepseek-7b")
        if adapter.detect():
            adapter.load()
            result = adapter.infer("你好")
            if result.success and result.structured:
                print(result.structured)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        super().__init__("local_llm")
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def detect(self) -> bool:
        """检测模型文件是否存在。不实际加载。"""
        if self.model_path and self.model_path.exists():
            self.status.available = True
            return True
        # 尝试常见路径
        common_paths = [
            Path("eve/core/deepseek-7b"),
            Path("eve/core/qwen"),
            Path("eve/core/llm"),
        ]
        for p in common_paths:
            if p.exists():
                self.model_path = p
                self.status.available = True
                return True
        self.status.last_error = "no model path found"
        return False

    def load(self, **kwargs) -> bool:
        """加载 LLM。

        如本地无依赖或显存不足，标记不可用并返回 False。
        """
        if not self.model_path or not self.model_path.exists():
            self.status.last_error = "model path not found"
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=kwargs.get("trust_remote_code", True)
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map=self.device if self.device == "cuda" else "cpu",
                torch_dtype="auto",
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )
            self.status.loaded = True
            self.status.device = str(self._model.device) if hasattr(self._model, "device") else self.device
            # 估算显存占用
            if self.device == "cuda" and torch.cuda.is_available():
                self.status.vram_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            return True
        except ImportError as e:
            self.status.last_error = f"dependency missing: {e}"
            return False
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                self.status.last_error = "CUDA out of memory — try cpu device"
            else:
                self.status.last_error = msg[:200]
            return False
        except Exception as e:
            self.status.last_error = str(e)[:200]
            return False

    def unload(self) -> None:
        """卸载模型释放资源。"""
        self._model = None
        self._tokenizer = None
        self.status.loaded = False
        self.status.vram_usage_mb = 0.0
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """执行 LLM 推理，自动解析 JSON 输出。

        参数:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数 (默认 2048)
            temperature: 温度 (默认 0.7)
            top_p: nucleus sampling (默认 0.9)
        """
        if not self._model or not self._tokenizer:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error="model not loaded — call load() first",
            )
        try:
            import torch

            t0 = time.perf_counter()
            inputs = self._tokenizer(prompt, return_tensors="pt")
            # 移动到模型所在设备
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # 只解码新生成的部分（跳过 prompt）
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            raw = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            elapsed = (time.perf_counter() - t0) * 1000
            self.status.last_inference_ms = elapsed

            tokens_used = len(generated_ids)
            structured = _parse_json_from_text(raw)

            return InferenceResult(
                success=True,
                output=raw,
                structured=structured,
                inference_time_ms=elapsed,
                tokens_used=tokens_used,
            )
        except Exception as e:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error=str(e),
            )


# ── VLM 适配器 ────────────────────────────────────────────

class VLMAdapter(BaseModelAdapter):
    """VLM（视觉语言模型）适配器。

    支持需要图像输入的多模态模型（如 Qwen-VL、CogVLM 等）。
    如果 VLM 不可用，可降级到 LLM + YOLO 组合。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        super().__init__("vlm")
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def detect(self) -> bool:
        """检测 VLM 模型文件是否存在。"""
        if self.model_path and self.model_path.exists():
            self.status.available = True
            return True
        common_paths = [
            Path("eve/core/qwen-vl"),
            Path("eve/core/vlm"),
        ]
        for p in common_paths:
            if p.exists():
                self.model_path = p
                self.status.available = True
                return True
        self.status.last_error = "no VLM model path found"
        return False

    def load(self, **kwargs) -> bool:
        """加载 VLM 模型。"""
        if not self.model_path or not self.model_path.exists():
            self.status.last_error = "VLM model path not found"
            return False
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path), trust_remote_code=kwargs.get("trust_remote_code", True)
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                str(self.model_path),
                device_map=self.device if self.device == "cuda" else "cpu",
                torch_dtype="auto",
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )
            self.status.loaded = True
            self.status.device = self.device
            return True
        except ImportError as e:
            self.status.last_error = f"dependency missing: {e}"
            return False
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                self.status.last_error = "CUDA out of memory — try cpu device"
            else:
                self.status.last_error = msg[:200]
            return False
        except Exception as e:
            self.status.last_error = str(e)[:200]
            return False

    def unload(self) -> None:
        """卸载 VLM 模型。"""
        self._model = None
        self._processor = None
        self.status.loaded = False
        self.status.vram_usage_mb = 0.0
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """执行 VLM 推理。

        kwargs:
            image: PIL Image 对象（必需）
            max_tokens: 最大 token 数
            temperature: 温度
        """
        image = kwargs.get("image", None)
        if image is None:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error="VLM infer requires 'image' kwarg (PIL Image)",
            )
        if not self._model or not self._processor:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error="VLM model not loaded — call load() first",
            )
        try:
            import torch

            t0 = time.perf_counter()
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 1024),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                )

            raw = self._processor.decode(outputs[0], skip_special_tokens=True)
            elapsed = (time.perf_counter() - t0) * 1000
            self.status.last_inference_ms = elapsed

            structured = _parse_json_from_text(raw)
            return InferenceResult(
                success=True,
                output=raw,
                structured=structured,
                inference_time_ms=elapsed,
                tokens_used=outputs.shape[1],
            )
        except Exception as e:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error=str(e),
            )


# ── YOLO 适配器 ───────────────────────────────────────────

class YOLOAdapter(BaseModelAdapter):
    """YOLO 目标检测适配器。

    封装 ultralytics YOLO，提供开关检测、加载、推理、结果查询。
    如果 YOLO 不可用，返回空结果而非假数据。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        super().__init__("yolo")
        self.model_path = model_path
        self.device = device
        self._model: Any = None

    def detect(self) -> bool:
        """检测 YOLO 依赖和模型文件是否可用。"""
        # 先检查 ultralytics 包
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.status.last_error = "ultralytics package not installed"
            return False

        # 检查模型文件
        if self.model_path:
            p = Path(self.model_path)
            if p.exists():
                self.status.available = True
                return True
            self.status.last_error = f"YOLO model not found: {self.model_path}"
            return False

        # 尝试常见路径
        common_paths = [
            Path("eve/core/yolo26/yolov8n.pt"),
            Path("eve/core/yolo26/best.pt"),
        ]
        for p in common_paths:
            if p.exists():
                self.model_path = str(p)
                self.status.available = True
                return True

        self.status.last_error = "no YOLO model found"
        return False

    def load(self, **kwargs) -> bool:
        """加载 YOLO 模型。"""
        if not self.model_path:
            self.status.last_error = "YOLO model path not set"
            return False
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            self.status.loaded = True
            self.status.device = self.device
            return True
        except ImportError:
            self.status.last_error = "ultralytics not installed"
            return False
        except Exception as e:
            self.status.last_error = str(e)[:200]
            return False

    def unload(self) -> None:
        """释放 YOLO 模型。"""
        self._model = None
        self.status.loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def detect_objects(
        self,
        image,
        conf_threshold: float = 0.5,
        **kwargs,
    ) -> DetectionResult:
        """对单张图像执行目标检测。

        Args:
            image: PIL Image 或 numpy array 或图片路径 str
            conf_threshold: 置信度阈值

        Returns:
            DetectionResult 包含检测到的对象列表
        """
        if not self._model:
            return DetectionResult(
                success=False,
                error="YOLO model not loaded — call load() first",
            )

        try:
            t0 = time.perf_counter()
            results = self._model(image, conf=conf_threshold, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            self.status.last_inference_ms = elapsed

            detections: list[ObjectDetection] = []
            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                cls_ids = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(cls_ids[i])
                    class_name = self._model.names.get(class_id, f"cls_{class_id}")
                    detections.append(ObjectDetection(
                        class_name=class_name,
                        confidence=float(confs[i]),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=class_id,
                    ))

            img_w = results[0].orig_shape[1] if results and hasattr(results[0], "orig_shape") else 0
            img_h = results[0].orig_shape[0] if results and hasattr(results[0], "orig_shape") else 0

            return DetectionResult(
                success=True,
                detections=detections,
                inference_time_ms=elapsed,
                image_size=(img_w, img_h),
            )

        except Exception as e:
            return DetectionResult(
                success=False,
                error=str(e),
            )

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """YOLO 不直接支持文本推理，这里提供兼容接口。

        作为 BaseModelAdapter 的抽象方法实现，实际调用 detect_objects。
        """
        image = kwargs.get("image", None)
        if image is None:
            return InferenceResult(
                success=False,
                output="",
                structured=None,
                inference_time_ms=0,
                error="YOLO infer requires 'image' kwarg. Use detect_objects() instead.",
            )
        det_result = self.detect_objects(image, **kwargs)
        return InferenceResult(
            success=det_result.success,
            output=json.dumps([{
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            } for d in det_result.detections], ensure_ascii=False),
            structured={"detections": [
                {"class": d.class_name, "confidence": d.confidence, "bbox": list(d.bbox)}
                for d in det_result.detections
            ]},
            inference_time_ms=det_result.inference_time_ms,
            error=det_result.error,
        )


# ── 适配器工厂 ────────────────────────────────────────────

def create_llm(
    model_path: str | Path | None = None,
    device: str = "cuda",
    auto_detect: bool = True,
) -> LLMAdapter:
    """创建 LLM 适配器实例，可选自动检测。"""
    adapter = LLMAdapter(model_path=model_path, device=device)
    if auto_detect:
        adapter.detect()
    return adapter


def create_vlm(
    model_path: str | Path | None = None,
    device: str = "cuda",
    auto_detect: bool = True,
) -> VLMAdapter:
    """创建 VLM 适配器实例，可选自动检测。"""
    adapter = VLMAdapter(model_path=model_path, device=device)
    if auto_detect:
        adapter.detect()
    return adapter


def create_yolo(
    model_path: str | Path | None = None,
    device: str = "cuda",
    auto_detect: bool = True,
) -> YOLOAdapter:
    """创建 YOLO 适配器实例，可选自动检测。"""
    adapter = YOLOAdapter(model_path=model_path, device=device)
    if auto_detect:
        adapter.detect()
    return adapter
