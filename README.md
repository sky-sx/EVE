# EVE

EVE（Embodied Virtual Entity）是运行在个人电脑环境中的可成长数字生命体
原型。它持续接收屏幕、鼠标、键盘活动和文本输入，由本地 LLM、基础模型与
可加载 TNN 协作；Memory 保存可检查的运行事实，Dock 负责离线训练。
麦克风与注意力/焦点机制尚未实现。

EVE 不是固定流程 Agent，也不以商业产品、插件生态或企业部署为当前目标。
项目遵循“非必须不设计”，优先保证可观察、可复现和可急停。

## 文档事实源

- `README.md`：项目入口与仓库边界。
- `EVE完整架构描述.txt`：用户确认的目标语义原文。
- `ARCHITECTURE.md`：当前结构化架构约束。
- `DECISIONS.md`：不得静默推翻的决定。
- `STATUS.md`：代码当前真实具备的能力。
- `ROADMAP.md`：尚未实现和仍待实验决定的事项。

目标架构与当前实现必须分开阅读。代码存在、单元测试通过、Mock 闭环、真实
输入、真实输出和真实成长是不同证据等级，不得互相替代。

## 仓库边界

```text
eve/                         当前正式 EVE 代码
tests/                       当前第一方测试
runs/                        运行产物，不是正式初始数据
datasets/                    通用数据目录，不预置实验数据
reference/                   外部参考工程，不是 EVE 实现
oldsrc/                      旧实现参考，不是正式代码
eve/core/yolo26/             内嵌第三方模型仓库
eve/core/qwen/               本地模型资源
eve/core/deepseek-7b/        本地模型资源
```

正式事实源只看 `eve/`。参考目录和内嵌第三方仓库不作为 EVE 当前能力证据。

## 运行

安全 smoke 不会执行真实键鼠或语音动作：

```powershell
python -m eve.main --profile smoke --duration 1 --run-dir runs/smoke
```

正式 GUI 入口：

```powershell
python -m eve.main --profile control
```

运行测试：

```powershell
python -m pytest -q
```

真实输出必须经过 Core 动作边界的权限、急停、接管和有效期检查，并由
Output worker 在执行前复检。Mock 结果不能表述为真实桌面动作，规则占位
节点不能表述为训练得到的 TNN。

## 开发原则

- 快速输入和动作路径不得等待 LLM、Memory 视图操作或 Dock 训练。
- 一个主要本地 LLM 承担日常通用的 LLM-based self update loop，不展示或
  保存隐藏推理。
- TNN 始终指 Tiny Neural Network；LLM、VLM 和 Memory 检索不改名为 TNN。
- Dock 和 Core 不内置固定任务、类别、教师或网络模板。
- 结果进入 Memory 后，以独立 MemoryUnit 和 Event 关联组织。
- 任何“已完成”声明必须在 `STATUS.md` 中给出匹配强度的证据。
