# 掩模圈地对抗场

Python 3.10+。原有本地裁判和示例 Bot 只使用标准库；强化学习模型、训练和评测额外需要 PyTorch 和 NumPy。解压后在包含 `arena.py` 的目录运行。Windows 也可以将命令中的 `python` 换成 `py -3`。

## 先看一局

双击 `example_replay.html` 即可在桌面浏览器离线查看已经跑好的 Bot1 对 Bot2。点击播放、单步、拖动进度条，或跳到下一次闭合、报废或重启。每帧是一个完整回合结算后的状态。事件栏标出发生在第一个移动步、第二个移动步还是回合末。

```bash
python arena.py play --seed 11 --open
python arena.py play --seed 11 --swap --out swapped.html
```

`play` 先跑完一局，随后写入独立 HTML。`--open` 会在本机默认浏览器打开。交换出生点通过交换 Bot 与 0/1 号方的绑定实现，朝向始终跟随对应方。

## 更换自己的 Bot1

直接修改 `bot1.py` 中的 `Bot.choose(self, state)`，或复制为 `my_bot.py` 后运行：

```bash
python arena.py play --bot1 my_bot --bot2 bot2 --seed 11 --open
python arena.py bench --bot1 my_bot --bot2 bot2 --seeds 11 29 71 --csv comparison.csv
python -m unittest -v test_arena
```

每个 Bot 都在独立的本地 Python 进程运行；同一局中对象可以保存计划，各局重新创建。初始方法 `Bot(board, side)` 得到静态地图和本方编号。双方在每个回合收到同一已结算局面的独立副本，同时选方向，返回单个字符串 U / R / D / L。收到的 `state.turn` 是已结算回合数，所以开局第一次调用时为 0。停机期间仍需返回方向。

| 数据 | 含义 |
|---|---|
| `side` | 自己的编号 0 或 1，对手编号为 `1-side` |
| `state.pos[i]`, `state.heading[i]` | 位置和朝向。停机位置 -1。朝向 0/1/2/3 对应 U/R/D/L |
| `state.owned[i]`, `state.trail[i]` | 领地、轮廓的独立整数位图 |
| `state.down[i]` | 剩余停机回合；0 表示在场 |
| `state.points[i]`, `state.leads[i]` | 当前得分、已领先回合数 |
| `board.defects`, `board.high`, `board.fast` | 缺陷、高分格、加速覆盖的位图 |
| `board.homes[i]`, `board.spawns[i]` | 校准区位图、出生点的格号 |
| `board.zones`, `board.markers` | 三个加速区各自的位图、对应标记点格号 |

格号为 `p = 41*y + x`。可用 `bool(state.owned[side] & (1 << p))` 检查某格归属，`engine.cells(mask)` 遍历其中的格号。对 `state.pos[i]` 做位运算前先检查是否在场。若想推演，调用 `trial = state.copy()`，再用 `engine.step(board, trial, [cmd0, cmd1])`。该函数修改传入副本，返回有序事件。

## 冻结的 Bot2 v1

- 从当前局面最多枚举 9 组双方方向，逐步执行与裁判共用的结算代码，检查加速中途碰撞、互切、重启和闭合。一般回合优先保留使自身报废的对手回应占比最低的走法；此占比用于排序，不代表对手的真实概率。
- 在己方边界生成带回程的不同宽度和深度的路线。对每条闭合候选执行板外扩散，分别重算双方分数，以分差增量评价中立地、高分地和敌方领地。根据完成回合数折算收益。
- 用含位置、朝向和加速步的广度优先搜索估计敌方最快触格时间，精确到回合内的移动步。它放宽未来轮廓和碰撞约束，并限制到 15 回合；超出范围统一截断。比较整条暴露轮廓与闭合时刻，对风险较大的长路线扣分。危险增大后重新寻找回家路线。
- 附近有敌线时，比较截线和敌方最早回家的时间；最后一回合直接比较对手不同回应下的终局分差和领先回合差。生成扩张路线留有软时间预算，正式的硬时限由裁判进程执行。
- Bot1 示例同样会寻找可完成的小圈，用于检查你的强化是否有效。Bot2 的候选以三段折线路径为主，遇到复杂凹形边界可能漏掉更优路线。短期枚举也无法保证多回合攻防的最优解。

## 如何评估版本

每个种子生成一张满足题述约束的地图，各打两局并交换出生点。在 CSV 中，所有带 1/2 后缀的统计列始终按 Bot1/Bot2 身份排列。字段 `win` 中 0 表示 Bot1 胜，1 表示 Bot2 胜，空值配合 `kind` 表示双败或待裁定。`us` 为累计回合请求耗时（微秒），`peak_us` 为单次最高耗时。优先看胜局数，再结合正常局平均分差、报废数、闭合次数、错误数和耗时分析原因。比较不同 Bot1 版本时用相同的 Bot2 文件、种子、机器与负载。另留一组种子供最后验证。

包内有两批实测原始记录。每批 3 张不同地图、每图 2 局。种子 11/29/71 上 Bot2 对示例 Bot1 为 4 胜 2 负，在另三张 101/202/303 上为 6 胜 0 负。12 局都正常结束。示例回放是种子 11 的一局。地图可复现；响应耗时和依赖软时间预算的搜索会随机器负载变化。

## 强化学习原型

这是可训练的实验原型，不附带训练好的模型，也不保证训练后优于 Bot2。训练和评测可以在另一台安装 Python 3.10+ 的机器上运行，先安装额外依赖：

```bash
python -m pip install -r requirements-rl.txt
```

安装后可运行 `python -m unittest -v` 检查裁判、模型、训练入口和评测入口。测试不训练模型。

训练器复用 `engine.step`，使用 PyTorch 实现的小型策略网络、自动微分与 REINFORCE（NumPy 用于特征处理）；对手按概率从 Bot2、冻结的历史模型和本局开始时冻结的当前模型中选择。两方基于同一个回合开始时的局面选动作，训练方轮换出生点。模型使用全局池化和局部精细视野，是有损压缩的局面表示。训练阶段不模拟进程通信、超时或累计响应时间；前两项判胜指标完全相同时按中性结果处理。训练可能耗时较长，实际吞吐取决于机器及 Bot2 对手比例。默认使用 CPU；可按训练机器的 PyTorch 安装情况选择 `--device cuda`、`--device cuda:0` 或 `--device mps`。

例如，以下命令在训练机器上运行 1000 局，地图种子范围为 0 至 99，输出到一个新目录：

```bash
python train.py --episodes 1000 --output-dir runs/experiment --train-seed-start 0 --train-seed-count 100 --seed 42 --checkpoint-every 100
```

训练会生成 `latest.pt`、定期编号的 `policy_*.pt`，以及记录训练地图种子范围的同名 `.json` 文件。使用 `--bot2-prob`、`--history-prob` 和 `--history-size` 调整对手池；历史池为空时，历史对手的概率暂时转给当前策略。默认奖励以终局胜负为主；`--score-weight` 非零会加入分差奖励，可能改变优化目标；`--gamma` 小于 1 会降低远期奖励权重。通过 `python train.py --help` 查看完整参数。

可从已有权重开始新一轮训练，输出目录必须与已有模型目录不同：

```bash
python train.py --resume runs/experiment/latest.pt --output-dir runs/continued --episodes 1000 --train-seed-start 100 --train-seed-count 100
```

续训需要权重旁的同名 JSON；它会保留已使用地图种子的记录，但会重新初始化随机数、历史对手池与基线，因此不是逐步精确恢复中断训练。

使用未参与训练的地图种子评测。以下示例分别对阵 Bot2 和一个历史模型，每张地图各打两局并交换出生点：

```bash
python evaluate.py --model runs/continued/latest.pt --opponent-model runs/experiment/policy_0000100.pt --seeds 1001 1002 1003 1004 1005 --output evaluation.json
```

评测使用 `arena.game` 的独立进程与 200 毫秒回合时限，记录逐局比分、胜负、错误和响应耗时。报告中的胜率以全部预定局数为分母，包含双败和待裁定局；样本量较小时不能据此认定稳定领先。若模型的 JSON 记录与评测种子重叠，评测会拒绝运行；缺失记录时会提示无法验证独立性，仍需自行保证地图未参与训练或模型选择。若需单独指定对手，可重复使用 `--opponent MODULE` 或 `--opponent-model PATH`；不指定模块时默认包含 Bot2。

也可以用已训练权重在原有回放入口中运行，运行前设置模型路径：

```bash
MASK_ARENA_MODEL=runs/continued/latest.pt python arena.py play --bot1 bot_rl --bot2 bot2 --seed 1001
```

Windows PowerShell 可先运行 `$env:MASK_ARENA_MODEL = 'runs/continued/latest.pt'`，再运行 `python arena.py play --bot1 bot_rl --bot2 bot2 --seed 1001`。`bot_rl` 按模型输出的概率采样动作；本地评测耗时受机器负载及 PyTorch 配置影响。旧版 NumPy `.npz` 权重可继续用于加载和续训，新保存的检查点使用 PyTorch `.pt` 格式。对接比赛仍需确认官方协议、依赖与资源限制，且官方地图可能与本地生成地图不同。

## 本地裁判范围

`engine.py` 处理棋盘与回合规则；`arena.py` 管理进程、同时请求、计时、技术判负和整场汇总。运行器测量本机进程通信的往返时间，时限为就绪 10 秒、开局 2 秒、回合 200 毫秒、结束 1 秒。超时或连接失败按满额累计回合耗时，非法的完整回复按实际耗时累计。第 10 次错误所在回合计时但不结算。技术判负导致的未结算回合不产生棋盘回放帧，终局栏显示最终错误数。全同待主办方裁定时，本地也保留待裁定。

地图按规则生成。这里没有官方地图、线上协议、提交构建系统或平台内存限制。模块加载失败作为本地就绪失败处理。对接官方材料时，保留策略，让适配层把官方局面转换为这里的 Board / State 即可。请在同一台机器上只加载你信任的本地 Bot 模块。
