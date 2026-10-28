# geatpy-rs 2.8.0 性能与 GPU 支持方案

- 状态：**实施中**：WP1、WP2 已完成并验证；WP3、WP4 待做。开发分支 `2.8.0-gpu`。
- 来源：2026-10-11 提出、当时搁置的 GPU 方案。评审结论是 GPU 整体收益有限，2.8.0 改为**先把 CPU 内核做快，
  GPU 只做轻量的求值支持**，GPU 算子卸载暂缓（设计保留在附录 A）。
- 基线：geatpy-rs 2.7.3（`3f1b534`）。

## 0. 目标与边界

目标：

1. 修掉 `mutde` 的 O(N²) 实现问题。
2. 恢复多核并行（rayon），覆盖大种群下的热点算子。
3. `evalVars` / `aimFunc` 可以直接返回 PyTorch / CuPy 张量，方便用户把目标函数放到 GPU 上。
4. 与官方 2.7.0 二进制的一致性约定不变：确定性算子并行前后**逐位一致**，随机算子**分布一致**；
   `tests/test_core_parity*.py` 全部通过。
5. 发布形态不变：单一 `cp38-abi3` 轮子，不引入 CUDA 或 torch 依赖。

不做：

- Rust 内核里写 CUDA / wgpu。
- 2.8.0 内的 GPU 算子卸载（附录 A）与种群常驻 GPU 的算法模板（附录 B）。

## 1. 评审依据

### 1.1 端到端剖析（2.7.1 数据，2.7.3 未改变这些热点）

用 cProfile 统计典型算法的耗时分布。测试函数是廉价的 numpy 实现（DTLZ2 M=3 D=12、Rastrigin D=30），
因此内核占比偏高；真实问题的目标函数往往是仿真或模型计算，求值通常占 90% 以上。

| 场景 | 总耗时 | 目标函数求值 | Rust 内核 | Python 其它 | 主要热点 |
| --- | --- | --- | --- | --- | --- |
| NSGA2, N=100, 300 代 | 0.08 s | 12% | 52% | 36% | ndsortESS 24% |
| NSGA2, N=1000, 60 代 | 0.26 s | 2% | 81% | 17% | ndsortESS 66% |
| NSGA2, N=5000, 12 代 | 0.58 s | 1% | 82% | 17% | ndsortESS 74% |
| NSGA3, N=105, 300 代 | 0.10 s | 9% | 70% | 21% | refselect 35% |
| NSGA3, N=1035, 40 代 | 0.48 s | 1% | 92% | 7% | refselect 66% |
| DE/rand/1/bin, N=100, 500 代 | 0.12 s | 14% | 50% | 36% | mutde 32% |
| DE/rand/1/bin, N=10000, 20 代 | 2.74 s | 2% | 96% | 2% | **mutde 93%** |

### 1.2 单算子耗时（2.7.3，单线程，2026-10-19 实测）

ObjV 为 N×3 的近似球面前沿，Chrom 为 N×10 的 RI 染色体：

| N | `ndsortESS` | `mutde` | `cdist(ObjV, ObjV)` |
| --- | --- | --- | --- |
| 1000 | 2.3 ms | 2.4 ms | 8.1 ms |
| 5000 | 46 ms | 33 ms | 142 ms |
| 20000 | 674 ms | 572 ms | 12.9 s |

`cdist` 的 N×N 结果在 N=20000 时有 3.2 GB，耗时主要花在内存上；模板里 `cdist` 多用于 N×R（参考点）的规模，
上表最后一列只说明 N×N 用法本身不可扩展，GPU 也无济于事（结果仍需回传主机）。

### 1.3 结论

- 常规规模（N ≤ 1000）每代内核只有几毫秒，GPU 的启动与传输开销会抵消收益。
- N=5000 时 GPU 卸载 ndsort 估计能让端到端快 3～4 倍；但真实问题中求值占 90% 以上，内核再快，整体最多快约 10%，
  且 Python 层的 17%～36% 开销 GPU 无能为力。
- GPU 求值对性能本身没有贡献：用户今天就可以在 `evalVars` 里自行转换张量；geatpy 能做的是省掉这几行样板代码。
- 确定的收益在 CPU 侧：`mutde` 修复（大种群预计约 100 倍）与多核并行（8～16 核上数倍），对所有用户生效，
  不增加依赖，不需要维护两套实现。

## 2. 工作包

### WP1 `mutde` 的 O(N²) 修复（已完成）

- 原实现：每个个体都构造长度 N 的候选池（`(0..n).filter(..).collect()` + `swap_remove`），整体 O(N²)。
- 现实现：从 0..N 拒绝采样 3～5 个下标，均匀、互不相同、`rows == n` 时排除自身；候选用完（N 很小）时与官方 2.7.0
  一样把全部行（含自身）重新装回、继续不放回抽取（2026-10-20 审查前两版都退化为有放回抽样，与官方不同）。分布与
  从候选池抽取完全相同，只是随机序列不同（同种子结果与 2.7.x 不同）。`Mask_N` 矩阵路径同样受益。
- 结果：单次调用 N=5000 从 33 ms 降到 1.5 ms，N=20000 从 572 ms 降到 6.7 ms。
- 验证：`tests/test_core_perf.py` 用“第 k 行取 2^k”的输入精确还原供体下标，检查两个差分供体都互不相同、不含自身、
  均匀分布，`Mask_N` > N 时自身以约 2/N 的比例出现；`tests/test_core_parity_review4.py` 检查小种群补池后同一对供体
  互不相同。fz.py 原有的 mutde 用例全部给定下标、从不随机抽供体，测不到这里；现已加入随机供体用例（含小种群），
  补池修正前与官方有 8 处统计差异，修正后为 0。`classic.py` 见下文“WP1 + WP2 端到端结果”。

### WP2 rayon 确定性并行（已完成）

原则：**结果与线程数无关**。只拆分与切分方式无关的工作（每个任务算一行输出或一个候选），
归约（求和）在并行算完各项后按原顺序串行完成，所以任何线程数下都与串行逐位一致。

- 基础设施：`crates/geatpy_core/src/parallel.rs`。自建 rayon 线程池（不用全局池，便于改线程数）；
  线程数默认取 `GEATPY_NUM_THREADS`，否则为逻辑 CPU 数，任何请求都截到逻辑 CPU 数（超额只会更慢，`2**63` 这类值
  也不再溢出）；`ea.set_num_threads(n)` / `ea.get_num_threads()`（返回实际使用的值），`set_num_threads(1)` 完全串行，
  `None` / 0 恢复默认。`pool_for(work, min_work)` 按工作量决定是否并行；建池时不持锁，建池失败（线程数受
  `RLIMIT_NPROC` 等限制）会记住，之后直接串行，不再每次重建。
- fork 安全：用 `os.register_at_fork` 注册处理函数：fork 前拿住线程池的锁，fork 后父进程放锁；子进程清空槽位
  （继承来的线程池没有工作线程，直接泄漏）后放锁，第一次需要时新建。早先按进程号判断的做法有两个漏洞：另一线程
  建池时持锁，此时 fork 的子进程继承一把永远不会释放的锁；进程号被后代复用时会用上没有工作线程的池。两者都会让
  子进程永久卡死，现已不存在（`fk.py` 式的压力测试：旧实现 1521 次 fork 卡死 2 次，新实现 5051 次 0 次）。
  只经由 `os.fork`（含 multiprocessing 的 fork 方式）的 fork 有效；其他扩展在 C 层直接调用 `fork()` 不经过这些处理函数。
- 浮点环境：工作线程沿用建池线程的浮点控制位（FTZ/DAZ、舍入方式）。“与线程数无关、逐位一致”只在默认浮点环境下
  成立；用户代码打开 FTZ（例如 `torch.set_flush_denormal(True)`）后，含次正规数的输入在多线程下可能与单线程不同。
- 内存：`map_collect_init` 让 refselect 的关联每个工作分片只分配一次临时缓冲区；GD/IGD 只在走并行时收集每行的
  最近距离，串行时按原顺序直接累加；cdist 不再预先收集 B 的行。每个工作线程预留约 2 MB 栈和（glibc 下）64 MB
  malloc arena 的虚拟地址空间，在 `ulimit -v` 下请设 `GEATPY_NUM_THREADS=1` 或 `MALLOC_ARENA_MAX=1`。
- 各算子（实测于 4 核 Xeon 2.1 GHz，与 2.7.3 对比）：

  | 算子 | 做法 | 2.7.3 | 1 线程 | 4 线程 |
  | --- | --- | --- | --- | --- |
  | `ndsortESS` / `TNS` / `DED`（N=20000, M=3） | 见下 | 655 ms | 172 ms | 64 ms |
  | 同上（N=50000, M=3） | | 4.2 s | 0.70 s | 0.28 s |
  | `refselect`（N=1035, M=3） | 关联按行并行；小生境选择改用有序集合与按参考点分组 | 28 ms | 6.0 ms | 2.1 ms |
  | `refselect`（N=10000, M=5） | | 2.7 s | 0.71 s | 0.17 s |
  | `refgselect`（N=10000, R=4950） | 余弦矩阵按行并行；按参考向量分组由每组扫描全体改为一次分桶 | 0.77 s | 0.71 s | 0.45 s |
  | `cdist`（2N×R, N=5050） | 按输出行并行 | 271 ms | 295 ms | 111 ms |
  | `IGD`（N=2000, PF≈10000） | 每行最近距离并行，再按原顺序求和（`GD`、`Spacing` 同） | 78 ms | 78 ms | 21 ms |

  - ndsort：原实现每层对每个未分层个体都扫描它之前的全部个体（O(层数·N²)）。现在每层维护前沿列表，只与前沿成员比较
    （单线程即快 3～6 倍）。M=2 时同一层前沿的第二个目标严格递减，只需与最后一个成员比较，始终串行（审查前合并 master
    时丢掉了这一点，大的二目标单前沿比 master 慢 6～45 倍）。M≥3 时按块处理：当前前沿已经较大（前沿大小 × 比较的目标数
    ≥ 8192，块至少 256 行）的块分两步并行：先剔除被块之前的前沿支配的候选，再让每个幸存者只与块内排在它前面的幸存者比较。被剔除的行
    若支配某个候选，由传递性该候选也被前沿支配，所以这与串行得到的前沿完全相同，比较次数也相同（证明写在
    `front_levels` 的注释里）。审查前的做法是按前若干个体外推前沿大小并让候选扫描块内全部未分层个体，对随机 M≥3
    种群会高估前沿、多做 3～13 倍比较，默认线程数下反而比单线程慢。计算期间释放 GIL。
  - `cdist` 的 N×N 用法主要耗在分配和写入结果矩阵（内存带宽），并行只有约 2 倍。
  - 阈值：每次并行调用有几十微秒的固定交接开销，阈值按“刚好开始划算”的总工作量设定：cdist 为 2^18，GD/IGD 为 2^16，
    Spacing 为 2^19（串行按对称只算一半距离，并行逐行全算），refgselect 的余弦矩阵（其后的关联和选择仍串行）为 2^17，
    refselect 为 2^16。审查前统一用 2^16，RVEA 常用规模的 refgselect、每代日志里的 IGD 等小调用在默认线程数下比单线程
    慢 1.2～2 倍。
  - 2026-10-28 按 master 提速后的串行路径重调：原阈值（前沿 2048、GD/IGD/Spacing 2^18、refgselect 2^20、
    refselect 2^17）下，中等规模 ndsort 并行反而更慢（1000×8 0.26 → 0.71 ms，4000×5 1.40 → 2.38 ms），Spacing 200×8
    40 → 47 µs，GD/IGD、refselect、refgselect 则错过收益（GD 200×3 串行 70 µs、并行 32 µs）。三目标 ndsort 并行从不
    划算（16000×3 10.4 / 10.5 ms），新阈值下保持串行。385 个规模的网格上，除 GD 102×8（19 → 23 µs）外默认线程数都
    不比单线程慢 10% 以上；GD/IGD/cdist 中位数为单线程的 0.29～0.34，refselect 0.54，refgselect 0.82。
  - 2026-10-20 审查修复后的实测（Apple M 系列 10 核，单位 ms，1 线程 / 默认）：

    | 场景 | 修复前 | 修复后 | master |
    | --- | --- | --- | --- |
    | ndsortESS，M=2 单前沿 N=20000 | 158 / 34 | 0.62 / 0.60 | 0.57 |
    | ndsortESS，M=3 随机 N=20000 | 19.6 / 43.9 | 18.8 / 18.8 | 159 |
    | ndsortESS，M=3 球面单前沿 N=20000 | 656 / 134 | 652 / 124 | 644 |
    | ndsortESS，M=5 随机 N=10000 | 34.5 / 16.3 | 34.0 / 20.3 | 80 |
    | IGD，N=2000 对 PF 9870 | 35.8 / 7.1 | 35.3 / 5.3 | 35.6 |
- **随机算子保持串行，`Parallel` 参数仍不生效**（与原计划不同）：N=10000、D=30 时 `recsbx`、`mutpolyn`、`mutgau`、
  `recdis`、`mutde` 等单次只要 2～8 ms，并行最多省几毫秒，却要改变随机序列、增加按块派生种子的复杂度，不值得。
- 顺带修复：`to_f64_array2` / `to_i32_array2` 改为总返回行优先数组。此前 Fortran 顺序的输入（例如转置）会让依赖
  连续内存的代码 panic（2.7.3 的 `mutbin` 即如此），新代码的 `cdist` 也依赖这一点。
- 验证：
  - 用 2.7.3 与新版本（1 线程、4 线程）对同一批输入逐位比较，全部一致：ndsort 三个函数 3312 例（含重复行、NaN、inf、
    约束、maxormins、needNum / needLevel，最大 N=20000；其中 18 例含 NaN 的输入两边同样 panic，见下）、
    `refselect` 与 `cdist` 866 例、`GD` / `IGD` / `Spacing` 240 例、`refgselect` 150 例，以及参考点列数与目标数不同的 `refselect`。
  - `tests/test_core_perf.py` 在仓库测试中比较 1 / 3 / 4 线程逐位一致，并覆盖 fork 子进程与 Fortran 顺序输入。
  - ObjV 含 NaN 时 ndsort 会 panic（2.7.3 即有）的问题已在 master 修复（按原版走 numpy unique 与 T-ENS）。合并后含 NaN 的
    输入沿用 master 的串行实现，不含 NaN 的输入走 `front_levels`；`refselect` / `refgselect` 的关联改用 master 复现原版
    舍入的算术（行归一化 + FMA 点积），仍按行并行。合并后与 master 构建逐位比较（1 / 4 线程）全部一致：NaN、整数并列、
    重复极点、重复参考向量 720 例，`refselect` 与 `cdist` 866 例，指标 240 例，`refgselect` 150 例。
  - master 为复现原版舍入改用 FMA 点积（`mul_add`），而 x86_64 轮子不能假定 CPU 有 FMA，每次 `mul_add` 都是一次库函数调用，
    合并后 `refselect` 单线程一度慢了约 3 倍。现在 `utils::dot_fma_rows` 一次算一行与整个矩阵的点积，每行做一次运行时
    FMA 检测并进入以 `fma` 特性编译的循环（已同时修复到 master）；FMA 软硬件都只舍入一次，结果不变（与改动前的构建逐位
    比较一致；Rust 单元测试把实际走的那条路径与用有理数算出、每步只舍入一次的参考值逐位比较，56 例中 40 例不融合时
    结果不同；审查前的测试在有 FMA 的机器上只是硬件对硬件，且 CI 从未运行 Rust 单元测试，现已加入）。上表 `refselect` / `refgselect` 的新版本数字
    均在合并 master 并加上该修复之后测得。
  - `refgselect` 单线程几乎没有收益：耗时主要是 N×R 的余弦矩阵，计算量与 2.7.3 相同，分桶只省掉了分组扫描；而 master 为
    复现原版舍入多做了行归一化，抵掉了一部分。它的收益来自多线程（4 线程约 1.6 倍）。

### WP1 + WP2 端到端结果

`scripts/bench/bench_core.py`，4 核 Xeon 2.1 GHz，总耗时在 cProfile 下测得（Python 部分会被放慢），单位秒：

| 场景 | 2.7.3 | 1 线程 | 4 线程 |
| --- | --- | --- | --- |
| NSGA2, N=1000, 60 代 | 0.69 | 0.39 | 0.30 |
| NSGA2, N=5000, 12 代 | 1.08 | 0.55 | 0.48 |
| NSGA3, N=1035, 40 代 | 0.89 | 0.59 | 0.28 |
| NSGA3, N=5050, 10 代 | 3.56 | 1.42 | 0.54 |
| DE/rand/1/bin, N=10000, 20 代 | 3.36 | 0.60 | 0.60 |

N ≈ 100 的场景（NSGA2 / NSGA3 / DE，约 0.2 s）三者持平，差异在计时波动（约 ±30%）之内（末尾的 GD/IGD 调用会建立
线程池，开销可以忽略）。注意：早先的 `bench_core.py` 只统计名字含 `_geatpy_core.` 的函数，`indicator.GD/IGD/HV/Spacing`
被算进了 `other`，因而得出“NSGA2 N=5000 的瓶颈在 Python 层（`other` 约 55%）”的错误结论；其中约 44 个百分点是
`ea.optimize` 结束时的指标计算，主要是单线程的 HV。脚本已修正，HV 是下一个热点。

`scripts/parity/classic.py`（36 个用例，两边各 30 次，2.7.3 对新版本单线程）：没有 |z| > 3 的用例。只用到确定性算子与
未改动随机算子的用例两边结果完全相同（z = 0）；用到 `mutde` 的 DE 类用例随机序列不同，|z| 最大 1.33。

### WP3 GPU 求值支持（轻量，只改 Python 层）

- 新增 `geatpy/_device.py` 的 `to_numpy(x)`：按类型鸭子识别，不主动 import torch / cupy。
  torch 用 `.detach().cpu().numpy()`，cupy 用 `.get()`，CPU 上的 DLPack 对象用 `np.from_dlpack`，其余 `np.asarray`。
- `Problem.evaluation` 在调用 `evalVars` / `aimFunc` 之后，把 `pop.ObjV`、`pop.CV` 中的张量转回 numpy；
  `calReferObjV` 的返回值同样处理。这样 `Algorithm.call_aimFunc` 的格式检查无需改动。
  属于“有意更宽松”（原版报 `ObjV is illegal`），写入 AGENTS.md 差异清单。
- 修正 `Problem.evaluation` 的异常吞没：它用 `except RuntimeError` 判断“子类没有重写 aimFunc”，
  而 torch 的 CUDA 错误（显存不足、设备端断言）也是 `RuntimeError`，会被吞掉并误报
  “aimFunc 和 evalVars 必须至少重写一个”。改为检查方法是否被重写（构造参数传入或子类方法），真实异常原样抛出；
  两者都未重写时，报错信息保持不变。
- 2.8.0 **不加** `Problem(device=...)` 参数：用户在 `evalVars` 开头自行 `torch.as_tensor(Vars, device=...)`，
  一行即可，且 dtype、设备、多卡分配都由用户掌控。是否加该参数视反馈再定。
- 示例：`demo/gpu/`，一个用 PyTorch 写目标函数的单目标问题和一个多目标问题，`device='cpu'` 也能跑。
- 测试 `tests/test_device.py`（无需 GPU）：返回 torch(cpu) 张量、带 `__dlpack__` / `__array__` 的伪数组时的自动转换；
  `aimFunc` 写回张量；`evaluation` 不再吞掉真实 `RuntimeError`；未安装 torch 时 import geatpy 不受影响。
  CI 增加一个可选 job（ubuntu、Python 3.12，安装 CPU 版 torch）；没有 torch 的 job 跳过相关用例。

### WP4 基准、文档与发布

- `scripts/bench/bench_core.py`：复现第 1 节两张表，并对 N ∈ {1k, 5k, 20k} 比较 1 线程与全部线程；
  结果回填本文档，用于定阈值和写 README。
- README / README.en 同步更新：性能数字、多核并行与线程数设置、“目标函数放到 GPU 上”的用法；`history.txt`；
  版本号 2.8.0（commitizen 管理的三处）。
- AGENTS.md：目录加入 `geatpy/_device.py`；更新“`Parallel` 不生效”与“有意更宽松”两条；
  验证表加入“改了并行路径：多线程逐位比较”。

## 3. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 并行改变确定性算子结果 | 只用不改变运算顺序的并行（按行 / 按块，每个输出元素的计算仍串行）；测试逐位比较 |
| 并行破坏随机可复现性 | 随机算子全部串行，`Parallel` 不生效，随机序列与线程数无关；除 `mutde`（WP1）外与 2.7.3 相同 |
| 小规模反而变慢 | 每个算子按实测的盈亏点设阈值，ndsort 按实际前沿大小决定是否并行；`ea.set_num_threads(1)` 可完全关闭 |
| 与用户自己的多进程 / 多线程争抢核心 | 线程数截到逻辑 CPU 数；README 说明多进程批量运行时在每个进程中设 `GEATPY_NUM_THREADS=1`（或核数 / 进程数） |
| fork、地址空间限制、浮点环境 | `os.register_at_fork` 处理线程池；`ulimit -v` 下用 1 线程或 `MALLOC_ARENA_MAX=1`；逐位一致的保证限定在默认浮点环境 |
| `mutde` 随机序列变化 | 分布不变，fuzz 与 classic 验证；在 history 中说明同种子结果与 2.7.x 不同 |

## 4. 里程碑

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| M1 | WP1 `mutde` 修复 + WP3 GPU 求值 | 全部测试通过；fuzz / classic 无异常；DE N=10000 提速 ≥ 10 倍 |
| M2 | WP2 并行框架 + `refselect` / `cdist` 并行 + 基准脚本 | 多线程逐位一致；NSGA-III 大种群提速数据回填 |
| M3 | WP2 `ndsort` 并行；WP4 文档，发布 2.8.0（随机算子并行已取消，见 WP2） | 同上；README 双语同步 |

## 附录 A：GPU 算子卸载（暂缓）

重启条件：有用户在 N 上万的规模下使用，且 WP2 并行后内核仍是瓶颈；先在 CUDA 机器上跑基准确认收益。

设计要点（评审前的方案，供重启时参考）：

- `ea.gpu.enable(device='cuda', min_size=None)` 全局开启，默认关闭；在 `geatpy/core/__init__.py` 的分派层，
  只对“规整调用”（二维 float ndarray、参数合法、N ≥ 阈值）走 torch 实现，其余交给 Rust，保证报错和返回类型不变。
- 只卸载确定性的 O(N²) 算子：`ndsortESS` / `ndsortTNS`（分块支配矩阵 + 逐层剥离，fp64 下逐位一致）、
  `refselect` 的关联步骤（随机部分留在 Rust）。`cdist` 的 N×N 结果需回传主机，不卸载。
- MPS 不支持 fp64，只支持 CUDA / ROCm；正确性测试用 torch 的 CPU 设备与 Rust 对比，无卡 CI 也能覆盖。

## 附录 B：种群常驻 GPU 的算法模板（远期预研）

在 `geatpy.gpu` 下提供 NSGA-II / NSGA-III / DE（RI 编码）的 torch 版模板，`evalVars` 直接接收与返回张量，
整代不回主机；参考 EvoX、EvoTorch 的向量化写法。只有 N 上万、目标函数也在 GPU 上时才有明显收益，视需求再定。
