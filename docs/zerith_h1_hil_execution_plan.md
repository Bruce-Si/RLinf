# Zerith H1 Pro 真机 HIL RL 执行计划

> 分支：`feature/zerith-h1-hil`
> RLinf 基线：`main@60889fb0`
> Zerith 基线：`ZERITH_Pi_OPEN/pi0@7b5d29a`
> 更新时间：2026-09-22

本文把已经在 H1 Pro 上验证过的 Zerith π0 控制链路接入 RLinf，并逐步推进到真机人在环强化学习。现有 JAX π0、50 步 action chunk、RTC、相机服务和 H1 SDK 闭环视为已验证基线；本计划验证的是它们能否满足 RLinf 的 robot、task、observation、action、intervention、replay 和 learner contract。

最终系统总览见：[Zerith H1 Pro HIL RL 架构图](./zerith_h1_hil_architecture.drawio)。

架构图按职责域分为三列。RLinf 负责 `ZerithRealEnv`、rollout、reward、replay 和 learner；ZERITH/OpenPI Policy Adapter 负责 DataConfig、transforms、normalization、JAX π0、50-step action chunk 和 RTC；H1 Pro 负责相机、SDK 和真机执行。实线表示实时控制路径，虚线表示异步 episode/replay/policy-update 回流。

## 1. 最终架构

RLinf 功能分支负责训练和实验编排，Zerith `pi0` 分支继续提供硬件和 JAX baseline：

```text
实时控制链：
H1 Camera/State
  → ZerithRealEnv
  → Zerith/OpenPI DataConfig
  → π0 Policy（H=50）
  → RTC + Safety Adapter
  → H1 SDK
  → H1 Pro

异步 HIL 回流：
H1 Pro
  → ZerithRealEnv（reward / success / timeout / intervention / executed action）
  → LeRobot Archive / Replay
  → Actor · Learner
  → policy update
  → π0 Policy
```

两条路径共享同一套 observation/action contract。RLinf 负责实时环境、reward、timeout、intervention、replay 和 learner；Zerith/OpenPI Policy Adapter 负责 DataConfig、transforms、normalization、π0、50-step action chunk 和 RTC。异步回流不阻塞实时控制链。

JAX baseline 可以绕过 RLinf：现有 `test_pi0.py` 直接驱动 JAX π0 + RTC，用于确认相机、状态、动作和 H1 控制闭环没有回归。RLinf HIL 路径通过 `ZerithRealEnv` 接入同一个 Policy Adapter；两条路径共享 DataConfig、normalization、action chunk 和 RTC 语义，不能各自维护一套。

动作边界固定为：

- 模型内部：32 维；
- RLinf task/environment：23 维；
- H1 物理 SDK：21 维；
- 维度 21、22 是保留的底盘维度，必须始终清零（现在固定底盘不动）；
- 发送给 H1 前检查底盘维度并裁剪为前 21 维；
- π0 action horizon 保持 50；
- 第一版建议 RLinf 每次执行窗口为 10 步，不能因此把模型输出误改成 10 步；
- RTC 位于 raw policy action 与 H1 SDK 之间。

## 2. 当前代码归属

### Zerith `pi0` 分支继续复用

| 文件 | 用途 | 适配方式 |
|---|---|---|
| `robot_infer/utils/real_env_sdk.py` | H1 SDK、23 维 qpos、相机服务、21 维下发 | 抽为 RLinf backend 的底层 wrapper |
| `robot_infer/scripts/test_pi0.py` | 已验证 JAX 真机 baseline | 保留作 baseline/parity 工具 |
| `robot_infer/utils/async_infer.py` | 异步推理和 action queue | 迁移调度语义，不复制 RLinf env |
| `robot_infer/utils/rtc_client.py` | RTC 服务通信 | 封装为执行 adapter |
| `src/openpi/policies/rtc_policy.py`、`src/openpi/models/rtc.py` | RTC 逻辑和 guidance | 第一阶段不改；全量 RL 前单独处理 log-prob |
| `src/openpi/training/config.py` | `pi0_zerith` 和 Zerith DataConfig | 作为 Zerith/OpenPI Policy Adapter 的行为基线；RLinf 只实现兼容 adapter，不把 DataConfig 当作 RLinf 训练核心 |
| `src/openpi/transforms.py` | 23/32 维、norm、delta/absolute、图像处理 | 以 golden test 为准迁移 |

### RLinf 使用的已有入口

- `rlinf/robotics/robot.py`、`rlinf/robotics/parts/base.py`：机器人组合和生命周期；
- `rlinf/robotics/discovery/registry.py`：RobotConfig、发现和 RobotInfo；
- `rlinf/envs/real/env.py`：真实环境、obs 转换、`chunk_step()`、timeout、intervention、metrics；
- `rlinf/envs/real/registry.py`：Gymnasium task factory；
- `rlinf/envs/real/wrappers/`：teleop、intervention、action transform 和归档；
- `rlinf/models/embodiment/openpi/openpi_action_model.py`：PyTorch π0 的 RL 接口；
- `rlinf/models/embodiment/openpi/`：RLinf 侧 policy adapter 接口；其中 Zerith DataConfig/transforms 是从 Zerith/OpenPI 行为迁移的兼容层，不是 RLinf 原生训练逻辑；
- `examples/embodiment/config/realworld_pnp_dagger_openpi.yaml`：真机 DAgger 模板；
- `rlinf/utils/ckpt_convertor/openpi/`：checkpoint 转换入口。

## 3. 分支和运行环境

分支已经创建：

```bash
cd /home/siwufei/work/chery/RLinf
git switch feature/zerith-h1-hil
```

现有未跟踪的 `model/` 目录不属于本适配工作，保持原样，不加入提交。

代码放在 RLinf 分支内，但 H1 SDK、camera service、Zerith OpenPI 依赖必须固定：

1. 记录 RLinf/Zerith commit、H1 SDK、camera service、CUDA 和 PyTorch 版本。
2. 使用独立虚拟环境，不依赖偶然的 `PYTHONPATH`。
3. H1 SDK 使用 lazy import，只在 `_open()` 或执行 adapter 初始化时导入。
4. mock、注册和 DataConfig 测试必须在没有 H1 SDK/相机的机器上运行。

## 4. 分阶段计划

### 阶段 0：冻结 contract 和 baseline

**目标**：把已验证链路写成可比较的事实。

建议新增：

```text
docs/zerith_h1_hil_contract.md
tests/zerith_fixtures/observation_*.npz
tests/zerith_fixtures/norm_stats/
```

记录 23 维 qpos/action 索引、单位、相机名称和顺序、RGB/BGR、分辨率、prompt、norm stats、horizon=50、query frequency、RTC 参数、gripper/head 规则、hold 行为，以及 baseline 成功率和延迟。

**验收**：固定 observation、prompt、norm stats 和初始 noise，可以重复运行 JAX baseline，并保存 raw action、RTC action、executed action 和 episode 结果。

依赖：无。不改运行代码。

### 阶段 1：注册 Zerith H1 robot backend

建议文件：

```text
rlinf/robotics/parts/zerith_h1.py
rlinf/robotics/parts/cameras/zerith_camera.py
rlinf/robotics/robots/zerith_h1.py
rlinf/robotics/robots/__init__.py
tests/unit_tests/test_zerith_h1_robot.py
tests/robot_mocks/zerith_h1.py
```

实现：

1. `ZerithH1Connection` 只保存地址、配置和 placement；构造不连接硬件。
2. `_open()` 创建 H1 SDK、camera client、watchdog/线程；`_release()` 先停动作，再停线程、deinit、断相机。
3. 双臂、waist、head 共用一个 SDK session 时，使用一个 owner connection 和 borrowed parts。
4. 声明稳定的 observation/action features；action contract 保持 23 维。
5. `send_action()` 检查 shape、dtype、有限值、chassis dims，再下发 action[:21]。
6. camera backend 通过 `Camera` contract 提供 `cam_high`、`cam_left_wrist`、`cam_right_wrist`；不假设 RealSense serial。
7. 增加 `ZerithH1Config`、`ZerithH1Robot.build()`、`Robot.register_type()` 和 discovery metadata。

配置初稿：

```yaml
type: ZerithH1
configs:
  - node_rank: 1
    camera_names: [cam_high, cam_left_wrist, cam_right_wrist]
    camera_grpc_target: localhost:50051
    h1_endpoint: ...
    disable_validate: false
```

**验收**：

- 无 H1 SDK 时可完成 registry 和 mock contract test；
- `Robot.of_type("ZerithH1", ...)`、`describe()`、connect/disconnect 可用；
- mock 下 observation 固定为 23 维，三路 frame 名称稳定；
- 错误 shape、NaN、非零 chassis 被拒绝；
- 相机断流进入明确的重连或安全保持路径；
- 真机只读连续 10 分钟，退出时 watchdog 生效。

依赖：阶段 0。

### 阶段 2：实现 `ZerithH1PickPlaceEnv`

建议文件：

```text
rlinf/envs/real/zerith_h1/base.py
rlinf/envs/real/zerith_h1/pick_and_place.py
rlinf/envs/real/zerith_h1/__init__.py
rlinf/envs/real/__init__.py
examples/embodiment/config/env/realworld_zerith_pick_place.yaml
tests/unit_tests/test_zerith_h1_task.py
```

任务环境负责 prompt、reset、成功/失败、reward、timeout 和 info；机器人连接、相机和动作由 backend 负责。

- 复用 `RealWorldEnv`、`register_tasks()`、wrapper stack 和 `chunk_step()`；
- action space 为 23 维，动作裁剪和 chassis 清零只能有一个明确执行点；
- reset 使用安全回零、等待操作者确认、清空 RTC queue；
- 第一版 reward 使用人工成功/失败标记或简单可靠的物体状态判定，不接大型 VLM reward model；
- 每个 step 记录 `executed_action`、`reward`、`success`、`timeout`、`intervene_flag`、`chunk_id` 和 timestamp，作为异步回流的输入；
- episode 结束时由 Archive writer 打包 observation、policy action、RTC action、human action 和结果；
- 注册 Gymnasium ID：`ZerithH1PickPlace-v1`。

**验收**：

- `import rlinf.envs.real` 后 registry 中出现该 ID；
- mock 下 `gym.make()`、reset、step、timeout、close 完成 episode；
- `RealWorldEnv` 得到 `states`、`main_images`、`extra_view_images`；
- `chunk_step()` 正确处理动作窗口、intervention 和终止；
- episode 结束后不会继续下发旧 chunk；
- mock 模式下可以走通 `H1 Pro → ZerithRealEnv → Archive → Learner → π0 Policy` 的异步回流，并验证 reward、intervention 和 `chunk_id` 对齐。

依赖：阶段 1。

### 阶段 3：冻结并接入 Zerith/OpenPI Policy Adapter

**归属原则**：DataConfig、transforms、normalization、camera mapping 和 50-step chunk 语义属于 Zerith/OpenPI Policy Adapter。RLinf 只定义 adapter 的调用接口和 batch/rollout contract，不重新定义这些机器人语义。

先以现有 Zerith JAX 实现为唯一行为基线，再决定是否在 RLinf 中实现兼容的 PyTorch transform。若只是 JAX baseline 真机验证，直接调用 Zerith adapter，不做 JAX checkpoint 转换。

建议文件：

```text
rlinf/models/embodiment/openpi/dataconfig/zerith.py
rlinf/models/embodiment/openpi/transforms/zerith.py
tests/unit_tests/test_zerith_openpi_transforms.py
tests/parity_tests/zerith_jax_torch_transform.py
```

输入 contract：

```text
RLinf states [B,23] + three frames + prompt
 -> Zerith repack/camera order
 -> pad state to [B,32]
 -> DeltaActions(mask=make_bool_mask(7,-1,7,-1,5,-2))
 -> ZeroDims(21,22)
 -> Normalize
 -> Resize/pad 224x224 and tokenize
 -> PyTorch π0
```

输出 contract：

```text
[B,50,32]
 -> model output transforms
 -> Unnormalize
 -> AbsoluteActions(same mask)
 -> ZeroDims(21,22)
 -> crop [B,50,23]
 -> RTC/execution adapter
```

必须单测 state/action 索引、camera 顺序和颜色、norm stats、delta mask、chassis zeroing、50 步输出、23 维裁剪、prompt，以及 transform 不被执行两次。

**验收**：用阶段 0 的真实 fixture，对 Zerith JAX adapter 与 RLinf 侧兼容 transform 的每个中间结果做 golden comparison；允许记录过的浮点误差，不能只比较最终成功率。还要验证 adapter 不会重复执行 normalization、23/32 padding 或 50-step chunk 展开。

依赖：阶段 0、2。

### 阶段 4：接入 JAX baseline

**目标**：不反向传播、不转换 checkpoint，验证 RLinf env、chunk、RTC、日志和归档。

建议文件：

```text
rlinf/models/embodiment/openpi/zerith_jax_adapter.py
rlinf/execution/zerith_rtc_adapter.py
tests/e2e_tests/realworld_zerith_jax_smoke.py
examples/embodiment/config/realworld_zerith_jax_eval.yaml
```

链路为：

```text
H1 camera/state -> ZerithRealEnv -> Zerith transform -> JAX policy/server
 -> [50,23] raw action -> RTC -> safety/watchdog -> H1
```

**验收**：20 个以上 mock episode、10 个短真机 episode；policy/RTC/executed action 具备 timestamp 和 chunk_id；无重复执行、跳步或旧 chunk；episode metrics、intervention 和 archive 可读取；与 `test_pi0.py` 的输入 transform 和物理动作一致。

依赖：阶段 1～3。

### 阶段 5：RLinf 原生 PyTorch π0 frozen rollout

新增/修改：

```text
rlinf/models/embodiment/openpi/dataconfig/__init__.py
examples/embodiment/config/model/pi0_zerith.yaml
examples/embodiment/config/realworld_zerith_frozen_eval.yaml
tests/parity_tests/zerith_pi0_model_parity.py
```

初始配置：

```yaml
config_name: pi0_zerith
action_dim: 32
action_horizon: 50
action_env_dim: 23
num_action_chunks: 10
train_expert_only: true
rtc_enabled: false
```

第一版在执行层接 RTC，避免模型和执行层重复 guidance。必须确认 RLinf 不会把 50 步 chunk 再次错误展开。

**验收**：固定输入加载 PyTorch policy，输出 [B,50,32]，transform 后 [B,50,23]；mock/真机 frozen rollout 均安全；记录与 JAX 的数值、延迟和频率差异。

依赖：阶段 3、4。

### 阶段 6：JAX checkpoint 转 PyTorch 并 parity

入口：`rlinf/utils/ckpt_convertor/openpi/jax_to_openpi_rlinf.py`。

核对 checkpoint pytree、`gemma_2b_lora` LoRA、`gemma_300m` action expert、32 维 projection、50 步 horizon、asset id/norm stats 和冻结参数。转换只负责模型权重，不负责 camera mapping、23/32 padding、normalization、RTC 或 H1 SDK。

使用同一 observation、prompt、norm stats 和同一初始 noise 比较：

1. model-space [50,32]；
2. unnormalize 后 [50,32]；
3. crop/zero 后 [50,23]；
4. RTC 输入和 physical [21]。

**验收**：至少三条真实 fixture 通过 parity 阈值；所有差异分类后才允许真机。

依赖：阶段 5。转换失败不能阻塞阶段 1～4。

### 阶段 7：HG-DAgger/HIL 数据闭环

复制：

```text
examples/embodiment/config/realworld_pnp_dagger_openpi.yaml
 -> examples/embodiment/config/realworld_zerith_pick_place_dagger_openpi.yaml
```

修改 task id、hardware type、camera、placement、`action_dim=32`、`action_env_dim=23`、horizon=50、`pi0_zerith`、norm stats、teleop、LeRobot robot_type/fps/path 和 RTC 参数。

每个 frame 保存：

```text
observation/state/images/timestamp
policy_action [50,23]
rtc_action
executed_action [21]
human/intervene_action
intervene_flag
episode_id/chunk_id
success/failure/timeout
```

**验收**：人工接管前后 action 来源可区分；flag 与接管一致；成功 episode 可归档和 replay；raw/RTC/human/executed action 按 chunk 对齐；至少 20 个 mock 流程和 10 个短真机 HIL episode。

依赖：阶段 2、4。阶段 5、6 是接入 RLinf 原生 PyTorch π0 的后续路线，不应阻塞基于已验证 JAX π0 的 HIL/DAgger。

### 阶段 8：冻结 Pi0 + residual/DSRL

冻结 Pi0，训练小型 residual policy 或 DSRL。先明确 residual 合并在 RTC 前还是后；replay 同时保存 base、residual、RTC、executed 和 intervention action；先离线/小步 update，再真机在线。

**验收**：一次 learner update 后可继续 rollout；residual=0 与 baseline 一致；边界和 watchdog 始终有效；intervention 不污染 replay；checkpoint 可回滚。

依赖：阶段 7。

### 阶段 9：LoRA/action expert/full Pi0 RL

顺序建议：

```text
LoRA Pi0 -> action expert -> 更大范围 Pi0 参数 -> full Pi0
```

进入前必须解决 RTC 后 executed action 与 log-prob 定义、50 步 chunk credit assignment、critic/reward 显存、rollout/learner 版本同步、真机失败恢复和安全停止。每次扩大训练范围都做 frozen baseline regression、动作边界检查、短真机 smoke test 和 checkpoint rollback。

## 5. 依赖、停止条件和提交拆分

依赖关系：

```text
阶段0 -> 阶段1 -> 阶段2 -> 阶段3 -> 阶段4 -> 阶段7 -> 阶段8 -> 阶段9
                               \-> 阶段5 -> 阶段6 --/
```

出现以下情况停止进入下一阶段：schema 不稳定；相机顺序不确定；raw/executed action 无时间戳对应；watchdog/急停未验证；episode 结束仍下发旧 chunk；transform parity 未分类；RTC 后动作和训练 log-prob 语义未确定却开始 PPO/full Pi0 RL。

阶段 4 完成后即可用 JAX π0 进入阶段 7 的真机 HIL；阶段 5、6 只为 RLinf 原生 PyTorch π0 frozen rollout 和 checkpoint parity 提供替代 policy 路线。阶段 7 使用哪条路线，由 `policy_backend=jax|torch` 配置明确选择。

建议提交拆分：

```text
1. docs + contract fixtures
2. H1 connection/parts/robot + mock
3. H1 task + Gym registration
4. camera service backend + read-only smoke
5. OpenPI DataConfig + golden tests
6. JAX baseline + RTC adapter
7. PyTorch pi0_zerith frozen rollout
8. checkpoint converter/parity
9. HIL/DAgger config and archive
10. residual RL
11. LoRA/action-expert/full Pi0 experiments
```

## 6. 启动命令形态

具体参数待实现后再核对，最终形态接近：

```bash
cd /home/siwufei/work/chery/RLinf
source .venv/bin/activate

python -m pytest \
  tests/unit_tests/test_zerith_h1_robot.py \
  tests/unit_tests/test_zerith_h1_task.py \
  tests/unit_tests/test_zerith_openpi_transforms.py

bash examples/embodiment/run_realworld_async.sh realworld_zerith_jax_eval
bash examples/embodiment/run_realworld_async.sh realworld_zerith_pick_place_dagger_openpi
```

真机命令前必须显式填写 H1 endpoint、camera service、RTC endpoint、norm stats、checkpoint、teleop 和 archive 路径，并确认 watchdog、急停和人工接管可用。

## 7. 工作量和算力

| 工作项 | 预计时间 |
|---|---:|
| contract/fixtures | 1～2 天 |
| robot backend/camera/mock | 3～5 天 |
| task/RealWorldEnv | 2～4 天 |
| Policy Adapter/DataConfig/golden | 3～5 天 |
| JAX baseline/RTC adapter | 2～3 天 |
| frozen PyTorch rollout | 3～5 天 |
| checkpoint conversion | 3～7 天 |
| HIL/DAgger | 3～5 天 |
| residual RL | 5～10 天 |

第一个真机 HIL 原型约 3～5 周，residual RL 再增加 1～2 周。前七阶段不需要四张 A100/H100；frozen rollout 和小型 residual RL 可先用一张较大显存 GPU。多卡主要在全量 Pi0、critic/reward model 和 rollout/learner 并行训练时才需要。

## 8. 本分支第一批实现范围

第一轮只实现：

```text
ZerithH1Connection / ZerithH1Robot
ZerithCamera
ZerithH1PickPlaceEnv
robot/task registration
mock/contract tests
```

第一轮不做 checkpoint conversion、full PyTorch Pi0 training、critic/reward model、PPO/full Pi0 RL。先通过 robot/task contract，再按阶段 3～9 推进。
