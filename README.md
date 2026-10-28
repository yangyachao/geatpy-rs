# **Geatpy (Rust Core Edition)**
The High-Performance Genetic and Evolutionary Algorithm Toolbox for Python, Powered by Rust.

[English](README.en.md) | **中文**

[![Release](https://github.com/yangyachao/geatpy/actions/workflows/release.yml/badge.svg)](https://github.com/yangyachao/geatpy/actions/workflows/release.yml)
[![Release](https://img.shields.io/github/v/release/yangyachao/geatpy?color=blue&logo=github)](https://github.com/yangyachao/geatpy/releases)
[![Python Version](https://img.shields.io/badge/python-3.8_~_3.15-blue.svg?logo=python&logoColor=white)](https://pypi.org/project/geatpy-rs/)
[![Rust](https://img.shields.io/badge/rust-PyO3_abi3-orange.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-green.svg)](LICENSE)

---

## 📌 项目渊源与致敬 (Upstream Heritage & Fork Notice)

本项目是经典高性能进化算法工具箱 **[geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)** 的现代化高性能 Rust 重构版本。

- **原官方项目主页**: [https://github.com/geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)
- **官方文档与教程**: [http://www.geatpy.com](http://www.geatpy.com) / [Geatpy Documentation](https://github.com/geatpy-dev/geatpy/tree/master/docs)
- **开源致谢**: 衷心感谢原作者团队在进化计算领域的杰出贡献与算法设计奠基。

### 为什么发起这个重构版本？(Motivation)
原版 Geatpy 的内核只以预编译二进制分发，在现代开发与生产环境中面临三个痛点：
1. **高版本 Python 无法安装**: 原版只发布到 Python 3.10 的预编译轮子。在 Python 3.11 ~ 3.15 上 pip 找不到匹配的轮子，只能退回源码构建，而原版内核并不随源码公开，基本无法安装。
2. **ARM 平台覆盖不全**: 原版的 Apple Silicon 轮子只覆盖 Python 3.8 ~ 3.10，且没有 Linux ARM64 (aarch64) 轮子，在 ARM 云服务器、树莓派与 Docker 容器里装不上。
3. **二进制内核难排错**: 长度为 1 的染色体、一维输入等边界情况会直接段错误，整个 Python 进程退出；还有多处行为与其自身文档不符。

为此，**本项目用 Rust (PyO3 + abi3) 重写了全部 74 个内核算子**，Python 层（算法模板、`Problem`、`Population` 等）沿用原版、接口不变
（只修正了原版的几处缺陷，并去掉了种群切片与合并时多余的数组复制），
提供 Python 3.8 ~ 3.15 通用的全平台预编译 Wheel。

---

## ⚡ 核心改进与特性对比 (Key Improvements)

| 特性维度 | 原官方版本 (Geatpy) | 本版本 (Geatpy Rust Core Edition) |
| :--- | :--- | :--- |
| **底层核心实现** | 编译好的二进制扩展，内核源码未公开 | **纯 Rust 原生实现** (`crates/geatpy_core`)，源码公开 |
| **Python 版本支持** | 仅支持 Python 3.5 ~ 3.10 | **Python 3.8 ~ 3.15 全覆盖支持**，另有 free-threaded（无 GIL）Python 3.14t 的 Wheel |
| **ABI 兼容性** | 每个 Python 小版本单独构建 | **PyO3 `abi3` 单一构建兼容所有 Python 3.8+ 版本**（free-threaded 版本没有稳定 ABI，单独构建） |
| **跨平台预编译** | Windows x86/x64、Linux x86_64、macOS（Apple Silicon 仅 Python 3.8 ~ 3.10），无 Linux ARM64 | **全平台官方预编译 Wheel (Linux x86_64/ARM64, macOS Universal2, Windows x64)** |
| **代码与内存安全** | 边界输入可能触发段错误，导致 Python 进程退出 | **安全 Rust（唯一的 `unsafe` 是确认 CPU 支持 FMA 后调用 FMA 指令版本的点积）；空、反向、超大等边界输入抛出 Python 异常而非段错误** |
| **多核并行与性能** | C OpenMP 并行 | 非支配排序、参考点选择、距离与 GD/IGD/Spacing 指标在大规模时自动多线程（结果与单线程逐位一致，`ea.set_num_threads()` / `GEATPY_NUM_THREADS` 设置线程数），其余算子单线程（`Parallel` 参数为兼容保留，不生效）；36 个经典用例耗时为原版的 0.05~0.76 倍；free-threaded Python 3.14t 上可以用线程同时运行多个模板（见常见问题） |
| **API 兼容度** | 原版基准 | **沿用原有 Python API、数据结构与 42 个算法模板；内核行为已与官方 2.7.0 二进制逐项对照验证（见下文）** |

> **多线程说明**：默认线程数为逻辑 CPU 数（设置的线程数也不会超过它）。同时运行多个 geatpy 进程时（批量任务、
> `multiprocessing`、`ProcessPoolExecutor`、joblib），请在每个进程中设 `GEATPY_NUM_THREADS=1` 或调用
> `ea.set_num_threads(1)`（或给每个进程“核数 / 进程数”个线程），否则线程争抢核心反而更慢。每个工作线程会预留约
> 66 MB 虚拟地址空间，在 `ulimit -v` 限制下请用 1 个线程或设 `MALLOC_ARENA_MAX=1`。“与单线程逐位一致”以默认浮点环境
> 为前提：若其他库打开了 flush-to-zero（如 `torch.set_flush_denormal(True)`），含次正规数的输入在多线程下可能不同。

### 大规模问题（十万变量）

每代耗时与进程峰值内存（Apple M2 Pro；目标函数为 numpy 向量化的 Sphere、ZDT1 与 0-1 背包，前者几乎不占时间；
单目标 20 代、NSGA2 10 代取平均，3 次运行的中位数；脚本 `scripts/parity/large_scale.py`）：

| 模板与问题 | 官方 2.7.0 | geatpy-rs 2.7.5 | 当前版本 |
| :--- | :--- | :--- | :--- |
| DE/rand/1/bin，Sphere，50 个体 | 87.4 ms，1028 MB | 43.4 ms，825 MB | **29.7 ms，824 MB** |
| DE/best/1/L，Sphere，50 个体 | 70.6 ms，1027 MB | 37.6 ms，825 MB | **23.6 ms，824 MB** |
| SEGA，实数编码，Sphere，50 个体 | 69.5 ms，800 MB | 34.4 ms，482 MB | **26.9 ms，482 MB** |
| EGA，实数编码，Sphere，50 个体 | 61.1 ms，496 MB | 30.3 ms，447 MB | **23.0 ms，445 MB** |
| SEGA，二进制编码，0-1 背包，50 个体 | 94.8 ms，633 MB | 33.1 ms，408 MB | **25.3 ms，406 MB** |
| NSGA2，ZDT1，100 个体 | 254 ms，1480 MB | 138 ms，1166 MB | **118 ms，1166 MB** |
| SEGA，二进制编码，0-1 背包，**50 万变量** | 442 ms，2621 MB | 164 ms，1744 MB | **125 ms，1744 MB** |

同一种子下 2.7.5 与当前版本的结果相同。十万变量时每代只需几十毫秒，速度和内存都不是瓶颈；
进化算法在这种规模上难以收敛（[#281](https://github.com/geatpy-dev/geatpy/issues/281)、
[#295](https://github.com/geatpy-dev/geatpy/issues/295)）是算法本身的局限，需要结合问题结构（变量分组、降维、专用算子）
或改用传统优化方法。

---

## ✅ 与官方 2.7.0 内核的一致性 (Parity)

Rust 内核以官方 geatpy 2.7.0 的 `_core` 二进制为行为基准，通过黑盒差分测试持续验证：

- **基准数据**：`scripts/parity/run_original_core.sh` 在 Docker (linux/amd64 + Python 3.6) 中运行官方二进制，
  生成 `tests/fixtures/core_parity_orig_2_7_0.json`；
- **确定性算子**（非支配排序、拥挤距离、编码解码、适应度分配、参考点、指标、NSGA-III/RVEA 环境选择等）逐值比对；
- **随机算子**（交叉、变异、选择）按统计特征比对；与官方一致，随机数取自 numpy 全局随机状态，
  `np.random.seed` / `ea.optimize(seed=...)` 可完整复现一次运行；
- **算法模板**（39 个单/多目标模板）比对最终解质量：`GEATPY_PARITY_TEMPLATES=1 pytest tests/test_core_parity.py`；
- **逐模块复核**（`tests/test_core_parity_review.py`）：对全部内核算子逐一核对，补齐了返回类型
  （全整数变量时返回 `int32`）、参数校验、`crtidp` 约束处理等细节，期望值均取自官方二进制的实际输出；
- **随机差分与交叉审查**（`tests/test_core_parity_review2.py`、`scripts/parity/fuzz/`）：约 1500 个随机形状、参数组合与
  边界值用例同时在官方二进制与 Rust 内核上运行，逐项比对数值、dtype、返回结构与报错，随机算子比对统计分布；
  并由另一模型（Antigravity）独立交叉审查。整数输出与官方一致为 `int32`，参数校验与报错信息与官方一致；
- **经典用例对比**（`scripts/parity/classic.py`）：Ackley、Rosenbrock、Rastrigin、att48 TSP、ZDT、DTLZ、BNH 等
  36 个问题与算法模板组合，覆盖最大化、混合整数、带约束、奇数种群、排列编码、多种群与多染色体（psy）模板，
  两边各运行 30 次比较最优值或 IGD/HV 的分布。除多种群 `soea_multi_SEGA_templet` 外均无显著差异；该模板的
  迁移用 `Replacement=2`，原版此处淘汰的是最优个体，本项目按文档淘汰最差个体，解明显更好（关闭迁移后两边一致）；
- **边界输入复核**（`tests/test_core_parity_review4.py`）：空种群、空参考点、反向或三维数组、超大尺寸、NaN、
  错误的参数类型与关键字等情形逐一对照官方二进制，报错类型与信息与原版一致；原版在其中一些情形下崩溃或越界读取，
  本项目改为抛出 Python 异常。

内核行为依据官方文档、docstring 与上述差分测试结果实现。
官方二进制中有几处与其自身文档不符的缺陷，本项目按文档实现，测试中单独说明：
`ndsortDED` 分层错误、`mergecv` 跨行累加、`tcheby`/`pbi` 在 `idealPoint=None` 时返回理想点、
`refselect` 候选矩阵行序错位、`migrate` 择劣替换（`Replacement=2`）实际淘汰的是最优个体、
`mutgau`/`mutuni` 的 `Middle=True` 以 `(ub-lb)/2` 而非区间中点为中心、
`crtip` 在边界为小数时会生成越界整数、`migrate` 随机替换（`Replacement=1`）总是淘汰最后一个个体且
完全网状结构（`Structure=0`）会从自身迁入、`pbi` 对恰在理想点上的个体返回 NaN、`cdist` 对非连续数组
（如 `A[::-1]`）读错数据、`refselect` 在极点矩阵奇异（如重复极点）时因舍入未能识别而解出无意义的截距（本项目退回列最大值）、
`crtri` 与 `crtip` 一样在边界为小数时生成越界整数、`mutde` 把 F 列表中的 None 当作另一个因子（文档写明 None 表示随机生成）、
`mutde` 的 `Mask_N` 为 N 行矩阵时不排除个体自身、`refgselect` 不检查 CV 行数且会就地修改传入的 CV、
`refselect`/`refgselect` 在参考点集为空时仍给出选择、`mselecting` 只给一个都没选中的种群补选
（文档保证每个种群至少选 ceil(种群个体数×MSel) 个，[#330](https://github.com/geatpy-dev/geatpy/issues/330)）；此外官方二进制在染色体长度为 1 或输入一维时会
段错误（`xovsp`/`xovsh` 等），在空种群上抽取 `mutde` 供体时崩溃，`refgselect` 的 Gamma 过短、GD/IGD/HV 的输入为空时读取越界内存，
Rust 内核正常处理或抛出 Python 异常。

> **注意**：`soea_DE_currentToRand_1_templet` 传入 `F=[0.5, None]`。原版实际把 F2 固定为 0.5，本项目按文档随机生成，
> 因此该模板在 Rastrigin 等问题上收敛比 2.7.0 慢（30 维、50×1000 时中位最优值约 93 → 156）。需要原版行为时可在运行前设置 `algorithm.mutOper.F = [0.5, 0.5]`。

---

## 📦 安装指南 (Installation)

### 方式一：通过 pip 一键安装 (PyPI 推荐)

```bash
pip install geatpy-rs
```

> 💡 **提示**：分发包名为 `geatpy-rs`，在 Python 代码中完全无需改动，继续沿用熟悉的 **`import geatpy as ea`** 即可无缝调用！
>
> free-threaded（无 GIL）的 Python 3.14t 上同样 `pip install geatpy-rs`，pip 会自动选择 `cp314t` 的 Wheel。

> 🧪 **预览版**：2.8.0b2 是多线程预览版（不含 GPU），`pip install` 默认仍安装 2.7.6 稳定版，试用请加 `--pre`：
> `pip install --pre geatpy-rs`。

### 方式二：从 GitHub Releases 下载预编译 Wheel

前往 [Releases 页面](https://github.com/yangyachao/geatpy/releases) 下载对应系统的 Wheel 文件，或通过链接一键安装：

```bash
# Linux (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.6/geatpy_rs-2.7.6-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Linux (ARM64 / aarch64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.6/geatpy_rs-2.7.6-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# macOS (Universal2 - 支持 Apple Silicon M1/M2/M3/M4 以及 Intel Mac)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.6/geatpy_rs-2.7.6-cp38-abi3-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl

# Windows (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.6/geatpy_rs-2.7.6-cp38-abi3-win_amd64.whl
```

### 方式三：从源码本地构建

克隆仓库后，仅需 Rust 工具链与 `maturin` 即可一键构建安装：

```bash
git clone https://github.com/yangyachao/geatpy.git
cd geatpy

# 使用内置跨平台构建脚本
python scripts/build.py --install

# 或使用 maturin 直接安装至当前环境
pip install maturin
maturin develop --release
```

详细本地编译、跨平台交叉编译与单元测试说明，请参阅 [BUILD.md](BUILD.md)。

---

## 🚀 快速上手 (Quick Start)

原有代码和案例库无需进行任何修改，直接沿用原有接口即可：

### 示例：使用 NSGA-III 求解经典 DTLZ1 三目标优化问题

```python
import numpy as np
import geatpy as ea

class DTLZ1(ea.Problem):
    def __init__(self, M=3):
        name = 'DTLZ1'
        maxormins = [1] * M
        Dim = M + 4
        varTypes = [0] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        XM = Vars[:, (self.M - 1):]
        g = np.array([100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
        ones_metrix = np.ones((Vars.shape[0], 1))
        pop.ObjV = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:, :self.M - 1]]), 1)) * \
                   np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))

if __name__ == '__main__':
    problem = DTLZ1(M=3)
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=500,
        logTras=1
    )
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=True)
```

> 💡 运行时间较长时，可以给 `ea.optimize` 传 `progress=True`，在终端显示一行进度条（按 MAXGEN、MAXTIME、MAXEVALS 中
> 进展最快的一个计算，最多每 0.2 秒刷新一次，不影响结果）。

### 用 torch / GPU 加速目标函数

Rust 内核只在 CPU 上运行，不直接支持 GPU。目标函数较重时，可以在 `evalVars` 里用 PyTorch（或 CuPy）计算，
返回前转回 numpy 数组：

```python
import numpy as np
import torch
import geatpy as ea

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # Apple 芯片可用 'mps'

class Rastrigin(ea.Problem):
    def __init__(self, D=1000):
        ea.Problem.__init__(self, 'Rastrigin', 1, [1], D, [0] * D, [-5.12] * D, [5.12] * D)

    def evalVars(self, Vars):
        x = torch.as_tensor(Vars, device=dev, dtype=torch.float32)  # numpy → torch（CPU 或 GPU）
        f = 10 * self.Dim + (x ** 2 - 10 * torch.cos(2 * torch.pi * x)).sum(1, keepdim=True)
        return f.cpu().numpy().astype(np.float64)  # 交回内核前转回 float64 的 numpy 数组

if __name__ == '__main__':
    algorithm = ea.soea_DE_rand_1_bin_templet(Rastrigin(), ea.Population(Encoding='RI', NIND=5000),
                                              MAXGEN=50, logTras=0)
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, saveFlag=False)
```

实测（Rastrigin + DE/rand/1/bin，1000～5000 维、5000～20000 个体）：

- 只看求值，Colab Tesla T4 上 CUDA 比 numpy 快 13～16 倍，Apple M2 Pro 的 MPS 快 8～15 倍；但 torch 在 CPU 上用 float32
  已能拿到大部分收益（比 numpy 快 2～10 倍）。
- 整次运行只快约 25%～35%：像 Rastrigin 这样逐元素计算量小的函数，求值本来只占小头，其余时间在内核与 Python 层。
  目标函数是仿真、神经网络等重计算时，GPU 的收益才明显。
- 每代都要在 numpy 与 GPU 之间来回拷贝种群，个体数很少（几百以内）时 GPU 反而更慢；MPS 不支持 float64。

---

## ❓ 常见问题 (FAQ)

下面是上游 [geatpy issues](https://github.com/geatpy-dev/geatpy/issues) 里反复出现的用法问题，示例代码均已实测。

### 提示 “Could not find any feasible solution”，明明有可行解？

约束写在 `evalVars` 返回的 CV 矩阵里，**每一列一个约束，CV <= 0 才算满足**。常见错误是符号写反，或者等式约束没有容差
（浮点数几乎不可能恰好相等）。求解失败时 `ea.optimize` 会打印最后一代每个约束的违反个数和最小值，便于定位。

```python
def evalVars(self, Vars):
    x1, x2 = Vars[:, [0]], Vars[:, [1]]
    f = x1 ** 2 + x2 ** 2
    CV = np.hstack([
        1 - x1 - x2,                     # x1 + x2 >= 1  写成  1 - x1 - x2 <= 0
        x1 - 2 * x2,                     # x1 <= 2 * x2
        np.abs(x1 - 0.3) - 1e-4,         # 等式 x1 == 0.3 要留容差
        np.minimum(x2 - 0.2, 0.8 - x2),  # “或”：x2 <= 0.2 或 x2 >= 0.8，取两者较小值
    ])
    return f, CV
```

### 怎样拿到每一代的结果、前 k 个解，或者接着上次的结果继续算？

```python
history = []
def outFunc(alg, pop):  # 每一代结束时调用，pop 是当代种群
    history.append((alg.currentGen, pop.ObjV.min()))

algorithm = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=100, outFunc=outFunc)
res = ea.optimize(algorithm, seed=1)

last = res['lastPop']                                       # 最后一代种群（多目标时 res['optPop'] 是非支配解集）
top = last[np.argsort(last.ObjV[:, 0] * problem.maxormins[0])[:10]]  # 单目标前 10 个解
print(top.Phen, top.ObjV)

# 继续优化：把上次的种群（或决策变量矩阵 res['Vars']）作为先验知识传入
algorithm2 = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=100)
res2 = ea.optimize(algorithm2, prophet=res['lastPop'])
```

想直观看到进化过程（例如旅行商路线逐代变短），可参考 `demo/soea_demo/soea_demo12`：用 `outFunc` 记录每一代的最优路线，
求解后保存为 GIF 动画（左边当代最优路线，右边收敛曲线）。

### 变量只能取某几个值、不能取某些值，或者要互不相等？

- 只能取一组给定的值（也可用来剔除某些值）：把变量设成整数下标，在 `evalVars` 里查表。
- 一组整数互不相等：用排列编码 `'P'`，变量个数可以少于取值范围。

```python
values = np.array([0.5, 1.2, 3.0, 7.5])  # 允许的取值

class MyProblem(ea.Problem):
    def __init__(self, Dim=3):
        # varTypes=1（整数），下标范围 0 ~ len(values)-1
        ea.Problem.__init__(self, 'Disc', 1, [1], Dim, [1] * Dim, [0] * Dim, [len(values) - 1] * Dim)

    def evalVars(self, Vars):
        X = values[Vars.astype(int)]
        return ((X - 3.1) ** 2).sum(1, keepdims=True)

# 5 个互不相等的整数，取自 0 ~ 19：种群用排列编码
# ea.soea_SEGA_templet(problem, ea.Population('P', NIND=40), MAXGEN=200)
```

### 目标函数需要额外的数据或参数，或者要在循环里换参数反复求解？

把数据存成问题类的属性：

```python
class Fit(ea.Problem):
    def __init__(self, data, weight):
        self.data, self.weight = data, weight
        ea.Problem.__init__(self, 'Fit', 1, [1], 2, [0, 0], [-10, -10], [10, 10])

    def evalVars(self, Vars):
        pred = Vars[:, [0]] * self.data[:, 0] + Vars[:, [1]]
        return ((pred - self.data[:, 1]) ** 2).mean(1, keepdims=True) + self.weight * (Vars ** 2).sum(1, keepdims=True)

for w in (0.0, 0.5):
    res = ea.optimize(ea.soea_DE_rand_1_bin_templet(Fit(data, w), ea.Population('RI', NIND=30), MAXGEN=200))
```

### 怎样设置交叉率、变异率等算子参数？

模板创建后修改它的 `recOper`（重组）和 `mutOper`（变异）：

```python
algorithm = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=200)
algorithm.recOper.XOVR = 0.9  # 交叉概率
algorithm.mutOper.Pm = 0.2    # 变异概率

de = ea.soea_DE_rand_1_bin_templet(problem, ea.Population('RI', NIND=50), MAXGEN=200)
de.mutOper.F = 0.7            # 差分进化的缩放因子
de.recOper.XOVR = 0.9         # 差分进化的交叉概率
```

### 能用多线程同时跑多个模板，或并行计算目标函数吗？

在 free-threaded（无 GIL）的 Python 3.14t 上可以：内核不需要 GIL，多个线程可以同时运行各自的模板，
或在 `evalVars` 里用线程池分块计算目标函数，不必像 `multiprocessing` 那样复制数据。

```python
from concurrent.futures import ThreadPoolExecutor
import geatpy as ea

def solve(_):
    problem = ea.benchmarks.Rastrigrin(30)
    algorithm = ea.soea_DE_rand_1_bin_templet(problem, ea.Population('RI', NIND=50), MAXGEN=500, logTras=0)
    res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    return res['ObjV'][0][0]

with ThreadPoolExecutor(8) as pool:
    best = list(pool.map(solve, range(8)))  # 同时运行 8 次
```

实测（Apple M2 Pro，8 个线程）：8 次独立运行时，DE 模板快 4.4 倍，目标函数用纯 Python 写的 SEGA 快 3.8 倍；
单次运行在 `evalVars` 里分块计算纯 Python 目标函数（400 个体 × 1000 变量），整体快 1.9 倍（其余时间花在算子与
数据转换上）。普通（有 GIL）的 Python 上线程没有加速。注意：

- 各线程共用 numpy 的全局随机状态，多线程同时运行时 `seed` 不能保证结果可复现；
- 内核自己的多线程（见上文“多线程说明”）会与这些线程争抢核心，同时运行多个模板时先调用 `ea.set_num_threads(1)`；
- 目标函数里对单个数逐个调用 numpy 函数（如对 Python float 调用 `np.cos`）时，线程之间会相互争用，
  线程越多反而越慢（8 个线程时只有单线程速度的 0.6 倍）；改用 `math` 模块，或对整个数组调用 numpy。

### 其它

- **每次运行结果都不一样**：传随机种子 `ea.optimize(algorithm, seed=1)`，同样的代码会得到完全相同的结果。
- **绘图与输出**：`drawing=0` 不画图，`1` 只画最终结果，`2` / `3` 实时画目标空间 / 决策空间；`drawLog=False` 不画迭代曲线；
  `saveFlag=False` 不保存结果文件，`dirName` 指定保存目录；`verbose=False`、`outputMsg=False` 关闭日志和结果输出。
- **目标函数很慢**：`evalVars` 一次拿到整个种群，尽量用 numpy 向量化；计算很重时可在其中用 `multiprocessing` 并行，
  或用 torch / GPU（见上文“用 torch / GPU 加速目标函数”）。

---

## 📊 架构设计与底层算子覆盖

Rust 内核位于 `crates/geatpy_core/`，导出 74 个算子，名称与原版一致：

- **编码与种群初始化 (`encoding.rs`)**: `crtfld`, `crtbp`, `crtip`, `crtrp`, `crtri`, `crtpp`, `crtpc`, `crtup`, `crtgp`, `bs2ri`, `bs2real`, `bs2int`, `ri2bs`, `boundfix`
- **选择与适应度 (`selection.rs`)**: `tour`, `etour`, `rws`, `sus`, `urs`, `rcs`, `rps`, `ecs`, `otos`, `dup`, `selecting`, `mselecting`, `ranking`, `scaling`, `powing`, `indexing`
- **交叉与重组 (`recombination.rs`)**: `xovsp`, `xovdp`, `xovud`, `xovsh`, `xovbd`, `xovexp`, `xovsec`, `xovpmx`, `xovox`, `recsbx`, `recdis`, `recint`, `reclin`, `recndx`, `recombin`
- **变异 (`mutation.rs`)**: `mutpolyn`, `mutgau`, `mutbga`, `mutuni`, `mutbin`, `mutde`, `mutinv`, `mutmove`, `mutswap`, `mutpp`, `mutate`
- **多目标、约束与迁移 (`multiobjective.rs`)**: `ndsortESS`, `ndsortTNS`, `ndsortDED`, `crowdis`, `cdist`, `mergecv`, `tcheby`, `pbi`, `crtidp`, `awGA`, `rwGA`, `refselect`, `refgselect`, `migrate`
- **性能指标 (`indicator.rs`)**: `GD`, `IGD`, `Spacing`, `HV`

---

## 📄 引用 (Citation) & 许可证 (License)

如果您在学术研究中使用了本项目，请按照原项目规范引用：

```bibtex
@misc{geatpy2020,
  author = {Jazzbin, J.},
  title = {Geatpy: The Genetic and Evolutionary Algorithm Toolbox with High Performance in Python},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/geatpy-dev/geatpy}}
}
```

本项目继承原项目协议，采用 **[GNU Lesser General Public License v3.0 (LGPL-3.0)](LICENSE)** 开源。

---

## 👤 维护者与版权 (Maintainer & Copyright)

- **Rust Core 发行版维护者**: yangyachao ([www.yangyachao.com](https://www.yangyachao.com))
- **联系邮箱**: [yangyachao@live.cn](mailto:yangyachao@live.cn)
- **代码仓库**: [https://github.com/yangyachao/geatpy](https://github.com/yangyachao/geatpy)

### 版权声明 (Copyright Notice)
- **Rust Core 重构实现与打包分发版权**: Copyright (C) 2026 yangyachao.
- **原版算法设计与 Python 架构版权**: Copyright (C) 2018-2021 Geatpy Development Team.
- 本项目基于 **[LGPL-3.0 License](LICENSE)** 协议开源。
