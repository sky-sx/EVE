# EVE

EVE（Embodied Virtual Entity）是一个运行在个人电脑环境中的可成长数字生命体原型。它持续接收屏幕、声音、鼠标、键盘和文本输入，由一个主要本地 LLM、必要时调用的云端 LLM、基础模型与可加载的 TNN 协作；Memory 保存各模块得到的结果及其关联；Dock 将缓慢、昂贵或反复出现的能力训练、压缩为新的 TNN。

EVE 不是固定流程 Agent，也不以商业产品、通用 Agent 框架、插件生态或企业级部署为当前目标。项目遵循“非必须不设计”，优先验证可观察、可复现且安全的成长闭环。

## 文档事实源

长期维护的 EVE 文档只有以下六份：

1. `README.md`：项目入口、文档边界和使用方式。
2. `EVE完整架构描述.txt`：用户确认的完整架构原文，是目标语义的最高事实源。
3. `ARCHITECTURE.md`：完整架构原文的结构化工程表达。
4. `DECISIONS.md`：已经确认、后续实现不得静默推翻的决定。
5. `STATUS.md`：当前代码实际做到了什么，以及证据强度。
6. `ROADMAP.md`：实现顺序、近期工作和仍待实验决定的问题。

`EXPERIMENT_RECORD_TEMPLATE.md` 是实验记录模板，不定义架构。

目标架构与当前实现必须分开阅读：`ARCHITECTURE.md` 描述 EVE 应当成为怎样的系统，`STATUS.md` 描述仓库今天真实具备的能力。代码存在、单元测试通过、Mock 闭环、真实输入、真实输出和真实成长是不同等级的证据，不得互相替代。

## 仓库边界

```text
eve/                         当前正式 EVE 代码
tests/                       当前第一方测试
runs/                        运行产物，不是文档事实源
datasets/                    实验数据
EVE完整架构描述.txt           完整架构原文
reference/                   外部参考工程，不是 EVE 实现
src/                         旧实现参考，不是正式代码
eve/core/yolo26/             内嵌第三方模型仓库
eve/core/qwen/               本地模型资源
eve/core/deepseek-7b/        本地模型资源
.trae/                       工具配置、技能与历史工作记录
```

`reference/`、`.trae/` 和内嵌模型仓库中的 Markdown 归各自工程或工具所有，不参与 EVE 架构合并，也不能作为 EVE 当前能力证据。它们只有在对应依赖、参考工程或工具本身不再需要时才应整体处置。

## 安全运行

正式入口当前只允许 `disabled` 和 `mock` 输出，不会通过命令行开启真实键鼠或语音动作：

```powershell
python -m eve.main --mode mock --smoke-seconds 1 --run-dir runs/smoke
```

运行测试：

```powershell
python -m pytest -q
```

真实输出必须经过 Safegate、用户明确授权和可用的急停路径。Mock 结果不能表述为真实桌面动作，规则占位节点不能表述为训练得到的 TNN。

## 开发原则

- 一次实验尽量只引入一个主要变量。
- 快速输入和动作路径不得等待 LLM、Memory 整理或 Dock 训练。
- 一个主要本地 LLM 承担日常通用思考，不预设固定多 Agent 角色体系。
- TNN 始终指 Tiny Neural Network；LLM、VLM 和 Memory 检索只共享节点运行语义，不因此改名为 TNN。
- 所有真实动作都必须经过 Safegate，用户接管与 Esc 急停优先。
- 结果进入 Memory 后，以独立 MemoryUnit 和关联组织，不把完整流程对象塞入每个最小记忆单元。
- 任何“已完成”声明必须在 `STATUS.md` 中给出与其强度匹配的证据。

