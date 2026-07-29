# EVE

EVE（Embodied Virtual Entity）是一个运行在个人电脑环境中的可成长数字生命体原型。它持续接收屏幕、鼠标、键盘活动和文本输入，由一个主要本地 LLM、必要时调用的云端 LLM、基础模型与可加载的 TNN 协作；Memory 保存各模块得到的结果及其关联；Dock 将缓慢、昂贵或反复出现的能力训练、压缩为新的 TNN。麦克风与注意力机制尚未实现。

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
oldsrc/                      旧实现参考，不是正式代码
eve/core/yolo26/             内嵌第三方模型仓库
eve/core/qwen/               本地模型资源
eve/core/deepseek-7b/        本地模型资源
.trae/                       工具配置、技能与历史工作记录
```

`reference/`、`.trae/` 和内嵌模型仓库中的 Markdown 归各自工程或工具所有，不参与 EVE 架构合并，也不能作为 EVE 当前能力证据。它们只有在对应依赖、参考工程或工具本身不再需要时才应整体处置。

## 安全运行

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

真实输出必须经过 Safegate、用户明确授权和可用的急停路径。Mock 结果不能表述为真实桌面动作，规则占位节点不能表述为训练得到的 TNN。

## 红蓝圆三角成长实验

实验分为两个阶段，使用同一套正式 EVE Core、Memory、Dock 和
Safegate，不在小游戏内泄漏目标坐标给 EVE。

### 阶段一：红圆单任务监督成长

```powershell
python -m experiments.red_blue_shapes --mode red_only `
  --run-dir runs/red_blue_phase1
```

1. 在控制窗口点击冷启动，确认屏幕、光标、键盘活动、Memory、
   Blackboard、LLM/VLM 和循环状态可见。
2. 只授权实验需要的鼠标移动与点击；Esc 始终作为急停。
3. 点击“请求 VLM 教师标签”，确认结果为绑定帧的结构化
   `red_circle / bbox / center / confidence`。
4. 人工点击红圆形成教师示范。每次示范保存屏幕、教师结果、动作、
   命中反馈、分数与时间组成的 Experience。
5. 点击“训练红圆 TNN”。若样本不足，TrainingOrder 保持
   `waiting_for_data`；累计到 20 个命中样本后 Dock 自动训练。
6. Dock 只用训练划分拟合，用独立留出集评价；通过门槛后保存
   artifact 并自动加载 `red_circle_locator`。
7. 让 EVE 在 Safegate 授权下自行点击，核对环境日志中的
   `source=eve`、命中率和 Memory 中完整 Experience。

阶段一验收结果应同时满足：红圆 TNN 已生成并加载、EVE 实际点击得分、
Memory 可检索训练经历、正常停机生成 `world.md` 与 `self.md`、重启后
恢复 TNN 配置。

### 阶段二：红蓝多任务适应且不遗忘

```powershell
python -m experiments.red_blue_shapes --mode instruction_driven `
  --run-dir runs/red_blue_phase2
```

1. 保留阶段一的 Memory 与红圆 artifact，输入任务：
   “保持红色圆形能力，同时点击蓝色三角形”。
2. 对蓝三角重复教师标签与示范采集；提交蓝三角 TrainingOrder。
3. Dock 生成独立的 `blue_triangle_locator`，不会覆盖红圆权重。
4. 两个 TNN 通过 SourceRef 同时消费屏幕，并可按各自频率运行；
   Core 分别维护 `available_tnn`、`active_tnn` 和 `loaded_tnn`。
5. 候选模型必须通过留出集；更新旧能力时还必须通过
   `regression_data` 回归集，否则标为 `candidate_rejected`，不替换已
   加载模型。
6. 分别统计红圆、蓝三角命中，并在正常停机和重启后再次验证两类任务。

阶段二同时支持 QNN 动作评价：

1. 在请求过 VLM 教师标签后，既采集命中点击，也主动采集未命中点击；
   环境分别记录 `reward=1` 和 `reward=-1`。
2. 累计至少 40 条正负 Experience 后点击“训练动作评价 QNN”。
3. Dock 训练屏幕状态＋动作到期望奖励的 critic；留出损失通过后自动加载
   `action_value_qnn`。
4. 多个动作 TNN 在同一轮产生候选时，QNN 逐一给出 `q_value`，Core 只把
   最高且超过最低分数的鼠标候选送入 Safegate。
5. 新动作 TNN 可在 TrainingOrder 中提供 `fitness_data`、
   `minimum_qnn_fitness` 和 `minimum_qnn_margin`。Dock 会用已加载 QNN
   评价候选及旧版本，未通过者不会替换现有 TNN。

当前不遗忘机制由“独立任务 TNN＋回归集＋QNN fitness margin”共同构成。
QNN 负责学习环境奖励和候选比较，不替代教师、不替代 Safegate，也不会
读取小游戏内部目标坐标。
物理鼠标、Esc、真实桌面 VLM 质量和最终命中率必须由用户在桌面会话中
授权并实机验收，自动测试不会代替这一步。

## 开发原则

- 一次实验尽量只引入一个主要变量。
- 快速输入和动作路径不得等待 LLM、Memory 整理或 Dock 训练。
- 一个主要本地 LLM 承担日常通用思考，不预设固定多 Agent 角色体系。
- TNN 始终指 Tiny Neural Network；LLM、VLM 和 Memory 检索只共享节点运行语义，不因此改名为 TNN。
- 所有真实动作都必须经过 Safegate，用户接管与 Esc 急停优先。
- 结果进入 Memory 后，以独立 MemoryUnit 和关联组织，不把完整流程对象塞入每个最小记忆单元。
- 任何“已完成”声明必须在 `STATUS.md` 中给出与其强度匹配的证据。
