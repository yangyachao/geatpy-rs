# **Geatpy (Rust Core Edition)**
The High-Performance Genetic and Evolutionary Algorithm Toolbox for Python, Powered by Rust.

**English** | [中文](README.md)

[![Release](https://github.com/yangyachao/geatpy/actions/workflows/release.yml/badge.svg)](https://github.com/yangyachao/geatpy/actions/workflows/release.yml)
[![Release](https://img.shields.io/github/v/release/yangyachao/geatpy?color=blue&logo=github)](https://github.com/yangyachao/geatpy/releases)
[![Python Version](https://img.shields.io/badge/python-3.8_~_3.15-blue.svg?logo=python&logoColor=white)](https://pypi.org/project/geatpy-rs/)
[![Rust](https://img.shields.io/badge/rust-PyO3_abi3-orange.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-green.svg)](LICENSE)

---

## 📌 Upstream Heritage & Fork Notice

This project is a Rust re-implementation of the core of **[geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)**,
a widely used evolutionary algorithm toolbox for Python.

- **Upstream project**: [https://github.com/geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)
- **Official docs and tutorials** (Chinese): [http://www.geatpy.com](http://www.geatpy.com) / [Geatpy Documentation](https://github.com/geatpy-dev/geatpy/tree/master/docs)
- **Acknowledgement**: many thanks to the original authors for their work on evolutionary computation and the design of the toolbox.

### Motivation
The core of the original Geatpy is distributed only as prebuilt binaries, which causes three problems today:
1. **No install on recent Python**: the original only ships wheels up to Python 3.10. On Python 3.11 – 3.15 pip finds
   no matching wheel and falls back to a source build, but the core's source is not published, so installation fails.
   The last upstream release (2.7.0) dates from January 2022.
2. **Incomplete ARM support**: Apple Silicon wheels exist only for Python 3.8 – 3.10, and there are no Linux ARM64
   (aarch64) wheels, so it cannot be installed on ARM cloud servers, Raspberry Pi or ARM Docker containers.
3. **Hard-to-debug binary core**: edge cases such as length-1 chromosomes or 1-D input crash the whole Python process
   with a segmentation fault, and several operators behave differently from their own documentation.

This project therefore **rewrites all 74 core operators in Rust (PyO3 + abi3)**, keeps the Python layer (algorithm
templates, `Problem`, `Population`, ...) and its interfaces (fixing only a few defects of the original and
dropping redundant array copies when populations are sliced or merged), and ships prebuilt wheels for every
platform that work on Python 3.8 – 3.15.

---

## ⚡ Key Improvements

| Aspect | Original Geatpy | Geatpy Rust Core Edition |
| :--- | :--- | :--- |
| **Core implementation** | Compiled binary extension; core source not published | **Pure Rust** (`crates/geatpy_core`), open source |
| **Python versions** | Python 3.5 – 3.10 only | **Python 3.8 – 3.15** |
| **ABI** | One build per Python minor version | **PyO3 `abi3`: a single build for every Python 3.8+** |
| **Prebuilt wheels** | Windows x86/x64, Linux x86_64, macOS (Apple Silicon only for Python 3.8 – 3.10); no Linux ARM64 | **Linux x86_64/ARM64, macOS Universal2, Windows x64** |
| **Memory safety** | Edge-case input can segfault and kill the Python process | **Safe Rust (the one `unsafe` call runs the FMA-instruction dot product after checking that the CPU has FMA); empty, reversed, oversized and other edge-case input raises a Python exception instead of crashing** |
| **Parallelism & speed** | C OpenMP | Non-dominated sorting, reference-point selection, distances and the GD/IGD/Spacing indicators use several threads on large inputs (bit-identical to one thread; set the count with `ea.set_num_threads()` / `GEATPY_NUM_THREADS`), the other operators run on one thread (the `Parallel` argument is accepted for compatibility but has no effect); 0.05 – 0.76× the original's runtime on 36 benchmark cases |
| **API compatibility** | Reference | **Same Python API, data structures and 42 algorithm templates; core behaviour verified against the official 2.7.0 binaries (see below)** |

> **Threads**: the default is the number of logical CPUs (a larger setting is capped at it). When several geatpy
> processes run at once (batch jobs, `multiprocessing`, `ProcessPoolExecutor`, joblib), set `GEATPY_NUM_THREADS=1` or
> call `ea.set_num_threads(1)` in each of them (or give each process cores / processes threads); otherwise the threads
> compete for the cores and run slower. Every worker thread reserves about 66 MB of virtual address space, so under
> `ulimit -v` use one thread or set `MALLOC_ARENA_MAX=1`. "Bit-identical to one thread" assumes the default
> floating-point environment: if another library turns on flush-to-zero (e.g. `torch.set_flush_denormal(True)`),
> input with subnormal numbers may give different results on several threads.

---

## ✅ Parity with the Official 2.7.0 Core

The official geatpy 2.7.0 `_core` binaries are the behavioural reference for the Rust core, verified continuously by
black-box differential testing:

- **Reference data**: `scripts/parity/run_original_core.sh` runs the official binaries in Docker
  (linux/amd64 + Python 3.6) and writes `tests/fixtures/core_parity_orig_2_7_0.json`;
- **Deterministic operators** (non-dominated sorting, crowding distance, encoding/decoding, fitness assignment,
  reference points, indicators, NSGA-III/RVEA environmental selection, ...) are compared value by value;
- **Random operators** (crossover, mutation, selection) are compared by their statistics. As in the original, random
  numbers come from numpy's global random state, so `np.random.seed` / `ea.optimize(seed=...)` reproduce a whole run;
- **Algorithm templates** (39 single- and multi-objective templates) are compared by final solution quality:
  `GEATPY_PARITY_TEMPLATES=1 pytest tests/test_core_parity.py`;
- **Operator-by-operator review** (`tests/test_core_parity_review.py`): every core operator was checked, which added
  return types (`int32` when all variables are integers), argument checks, `crtidp` constraint handling and more;
  every expected value is the official binaries' actual output;
- **Randomised differential testing and cross-review** (`tests/test_core_parity_review2.py`, `scripts/parity/fuzz/`):
  about 1500 cases with random shapes, argument combinations and edge values run on both the official binaries and the
  Rust core, comparing values, dtypes, return structures and errors (statistics for random operators); a second model
  (Antigravity) reviewed the core independently. Integer outputs are `int32` and argument checks and error messages
  match the original;
- **Benchmark comparison** (`scripts/parity/classic.py`): 36 problem/template combinations (Ackley, Rosenbrock,
  Rastrigin, att48 TSP, ZDT, DTLZ, BNH, ...) covering maximisation, mixed integers, constraints, odd population sizes,
  permutation encoding, multi-population and multi-chromosome (psy) templates, each run 30 times on both cores to
  compare the distribution of the best objective or of IGD/HV. No significant difference except for the
  multi-population `soea_multi_SEGA_templet`: its migration uses `Replacement=2`, where the original evicts the *best*
  individual; this project evicts the worst one as documented and finds clearly better solutions (with migration
  disabled both cores agree);
- **Edge-case review** (`tests/test_core_parity_review4.py`): empty populations, empty reference sets, reversed or 3-D
  arrays, oversized shapes, NaN, wrong argument types and keywords are checked against the official binaries, with the
  same exception types and messages; where the original crashes or reads out of bounds, this project raises a Python
  exception.

The core follows the official documentation, the docstrings and the differential test results above. Where the
official binaries contradict their own documentation, this project follows the documentation and the tests note it:
`ndsortDED` assigns wrong fronts, `mergecv` sums across rows, `tcheby`/`pbi` return the ideal point when
`idealPoint=None`, `refselect` misorders the candidate matrix, `migrate` with `Replacement=2` evicts the best individual,
`mutgau`/`mutuni` with `Middle=True` centre on `(ub-lb)/2` instead of the interval midpoint, `crtip` produces
out-of-range integers for fractional bounds, `migrate` with `Replacement=1` always evicts the last individual and the
fully connected structure (`Structure=0`) migrates from a population to itself, `pbi` returns NaN for an individual
exactly at the ideal point, `cdist` reads non-contiguous arrays (such as `A[::-1]`) incorrectly, `refselect` misses a singular extreme-point matrix (e.g. a repeated extreme point) through roundoff and solves it into meaningless intercepts (this project falls back to the column maxima),
`crtri` produces out-of-range integers for fractional bounds like `crtip`, `mutde` replaces a None in an F list with the
other factor (the documentation says None means random), `mutde` with an N-row `Mask_N` matrix does not exclude the
individual itself from the donors, `refgselect` does not check the rows of CV and modifies the caller's CV in place, and
`refselect`/`refgselect` still select individuals when there is no reference point, and `mselecting` only refills
populations that got no individual at all (the documentation guarantees ceil(size × MSel) per population,
[#330](https://github.com/geatpy-dev/geatpy/issues/330)). The official binaries also segfault
on length-1 chromosomes or 1-D input (`xovsp`/`xovsh`, ...), crash when `mutde` draws donors from an empty population,
and read out of bounds for a short `refgselect` Gamma or empty GD/IGD/HV input; the Rust core handles these or raises a
Python exception.

> **Note**: `soea_DE_currentToRand_1_templet` passes `F=[0.5, None]`. The original actually fixes F2 at 0.5; this project
> draws it at random as documented, so the template converges more slowly than 2.7.0 on Rastrigin and similar problems
> (30 variables, 50×1000: median best about 93 → 156). For the original behaviour set `algorithm.mutOper.F = [0.5, 0.5]`
> before running it.

---

## 📦 Installation

### Option 1: pip (PyPI, recommended)

```bash
pip install geatpy-rs
```

> 💡 **Note**: the distribution is named `geatpy-rs`, but nothing changes in your code: keep using **`import geatpy as ea`**.

> 🧪 **Preview**: 2.8.0b1 is the multithreading preview (no GPU support). `pip install` still installs the 2.7.5 stable
> release; add `--pre` to try it: `pip install --pre geatpy-rs`.

### Option 2: prebuilt wheels from GitHub Releases

Download the wheel for your platform from the [Releases page](https://github.com/yangyachao/geatpy/releases), or install it directly:

```bash
# Linux (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.5/geatpy_rs-2.7.5-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Linux (ARM64 / aarch64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.5/geatpy_rs-2.7.5-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# macOS (Universal2: Apple Silicon M1/M2/M3/M4 and Intel)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.5/geatpy_rs-2.7.5-cp38-abi3-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl

# Windows (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.5/geatpy_rs-2.7.5-cp38-abi3-win_amd64.whl
```

### Option 3: build from source

After cloning, only a Rust toolchain and `maturin` are needed:

```bash
git clone https://github.com/yangyachao/geatpy.git
cd geatpy

# cross-platform build script
python scripts/build.py --install

# or install into the current environment with maturin
pip install maturin
maturin develop --release
```

See [BUILD.md](BUILD.md) (Chinese) for local and cross-platform builds and for running the tests.

---

## 🚀 Quick Start

Existing code and examples work as they are:

### Example: NSGA-III on the three-objective DTLZ1 problem

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

> 💡 For long runs, pass `progress=True` to `ea.optimize` to show a one-line progress bar in the terminal (it follows
> whichever of MAXGEN, MAXTIME and MAXEVALS is closest to its limit, redraws at most every 0.2 s and does not change
> the results).

### Speeding up the objective function with torch / GPU

The Rust core runs on the CPU only and has no GPU support. When the objective function is expensive, compute it with
PyTorch (or CuPy) inside `evalVars` and convert the result back to a numpy array:

```python
import numpy as np
import torch
import geatpy as ea

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # on Apple silicon use 'mps'

class Rastrigin(ea.Problem):
    def __init__(self, D=1000):
        ea.Problem.__init__(self, 'Rastrigin', 1, [1], D, [0] * D, [-5.12] * D, [5.12] * D)

    def evalVars(self, Vars):
        x = torch.as_tensor(Vars, device=dev, dtype=torch.float32)  # numpy -> torch (CPU or GPU)
        f = 10 * self.Dim + (x ** 2 - 10 * torch.cos(2 * torch.pi * x)).sum(1, keepdim=True)
        return f.cpu().numpy().astype(np.float64)  # hand a float64 numpy array back to the core

if __name__ == '__main__':
    algorithm = ea.soea_DE_rand_1_bin_templet(Rastrigin(), ea.Population(Encoding='RI', NIND=5000),
                                              MAXGEN=50, logTras=0)
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, saveFlag=False)
```

Measured on Rastrigin with DE/rand/1/bin (1000 – 5000 variables, 5000 – 20000 individuals):

- The evaluation alone runs 13 – 16 times faster with CUDA on a Colab Tesla T4 and 8 – 15 times faster with MPS on an
  Apple M2 Pro than with numpy, but torch with float32 on the CPU already gets most of that (2 – 10 times faster).
- A whole run is only about 25% – 35% faster: for an element-wise objective like Rastrigin, evaluation is only a small
  part of the run and the rest is spent in the core and the Python layer. The GPU pays off when the objective is a
  heavy computation such as a simulation or a neural network.
- The population is copied between numpy and the GPU every generation, so with few individuals (a few hundred or
  fewer) the GPU is slower; MPS has no float64.

---

## ❓ FAQ

Usage questions that keep coming up in the upstream [geatpy issues](https://github.com/geatpy-dev/geatpy/issues).
Every snippet has been run.

### "Could not find any feasible solution", although there obviously is one?

Constraints go into the CV matrix returned by `evalVars`, **one column per constraint, and a constraint holds when
its CV is <= 0**. The usual mistakes are a flipped sign or an equality without a tolerance (floats are almost never
exactly equal). When no feasible solution is found, `ea.optimize` prints how many individuals of the last generation
violate each constraint and its smallest value.

```python
def evalVars(self, Vars):
    x1, x2 = Vars[:, [0]], Vars[:, [1]]
    f = x1 ** 2 + x2 ** 2
    CV = np.hstack([
        1 - x1 - x2,                     # x1 + x2 >= 1  becomes  1 - x1 - x2 <= 0
        x1 - 2 * x2,                     # x1 <= 2 * x2
        np.abs(x1 - 0.3) - 1e-4,         # an equality x1 == 0.3 needs a tolerance
        np.minimum(x2 - 0.2, 0.8 - x2),  # "or": x2 <= 0.2 or x2 >= 0.8, take the smaller one
    ])
    return f, CV
```

### How do I get every generation, the best k solutions, or continue from a previous run?

```python
history = []
def outFunc(alg, pop):  # called after every generation with the current population
    history.append((alg.currentGen, pop.ObjV.min()))

algorithm = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=100, outFunc=outFunc)
res = ea.optimize(algorithm, seed=1)

last = res['lastPop']                                       # last population (res['optPop'] is the Pareto set for MOEAs)
top = last[np.argsort(last.ObjV[:, 0] * problem.maxormins[0])[:10]]  # best 10 of a single-objective run
print(top.Phen, top.ObjV)

# continue: pass the previous population (or the decision matrix res['Vars']) as prior knowledge
algorithm2 = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=100)
res2 = ea.optimize(algorithm2, prophet=res['lastPop'])
```

To watch the evolution (for example a travelling-salesman tour getting shorter), see `demo/soea_demo/soea_demo12`: it
records the best tour of every generation with `outFunc` and saves a GIF (current best tour on the left, convergence
curve on the right).

### Variables that may only take some values, must skip some values, or must all differ?

- Only a given set of values (which also skips the others): make the variable an integer index and look it up in
  `evalVars`.
- Distinct integers: use the permutation encoding `'P'`; there may be fewer variables than values in the range.

```python
values = np.array([0.5, 1.2, 3.0, 7.5])  # allowed values

class MyProblem(ea.Problem):
    def __init__(self, Dim=3):
        # varTypes=1 (integer), indices 0 .. len(values)-1
        ea.Problem.__init__(self, 'Disc', 1, [1], Dim, [1] * Dim, [0] * Dim, [len(values) - 1] * Dim)

    def evalVars(self, Vars):
        X = values[Vars.astype(int)]
        return ((X - 3.1) ** 2).sum(1, keepdims=True)

# 5 distinct integers out of 0 .. 19: use a permutation-encoded population
# ea.soea_SEGA_templet(problem, ea.Population('P', NIND=40), MAXGEN=200)
```

### The objective needs extra data or parameters, or I solve it in a loop with different parameters?

Store them as attributes of the problem:

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

### How do I set the crossover and mutation rates?

Change the template's `recOper` (recombination) and `mutOper` (mutation) after creating it:

```python
algorithm = ea.soea_SEGA_templet(problem, ea.Population('RI', NIND=50), MAXGEN=200)
algorithm.recOper.XOVR = 0.9  # crossover probability
algorithm.mutOper.Pm = 0.2    # mutation probability

de = ea.soea_DE_rand_1_bin_templet(problem, ea.Population('RI', NIND=50), MAXGEN=200)
de.mutOper.F = 0.7            # DE scale factor
de.recOper.XOVR = 0.9         # DE crossover probability
```

### Other

- **Every run gives a different result**: pass a seed, `ea.optimize(algorithm, seed=1)`; the same code then gives
  exactly the same result.
- **Plots and output**: `drawing=0` draws nothing, `1` only the final result, `2` / `3` the objective / decision space
  live; `drawLog=False` skips the trace plots; `saveFlag=False` saves no files and `dirName` sets the folder;
  `verbose=False` and `outputMsg=False` turn off the log and the result summary.
- **The objective is slow**: `evalVars` receives the whole population, so vectorise with numpy; for heavy computations
  use `multiprocessing` inside it, or torch / a GPU (see "Speeding up the objective function with torch / GPU" above).

---

## 📊 Architecture and Operators

The Rust core lives in `crates/geatpy_core/` and exports 74 operators with the original names:

- **Encoding and population initialisation (`encoding.rs`)**: `crtfld`, `crtbp`, `crtip`, `crtrp`, `crtri`, `crtpp`, `crtpc`, `crtup`, `crtgp`, `bs2ri`, `bs2real`, `bs2int`, `ri2bs`, `boundfix`
- **Selection and fitness (`selection.rs`)**: `tour`, `etour`, `rws`, `sus`, `urs`, `rcs`, `rps`, `ecs`, `otos`, `dup`, `selecting`, `mselecting`, `ranking`, `scaling`, `powing`, `indexing`
- **Crossover and recombination (`recombination.rs`)**: `xovsp`, `xovdp`, `xovud`, `xovsh`, `xovbd`, `xovexp`, `xovsec`, `xovpmx`, `xovox`, `recsbx`, `recdis`, `recint`, `reclin`, `recndx`, `recombin`
- **Mutation (`mutation.rs`)**: `mutpolyn`, `mutgau`, `mutbga`, `mutuni`, `mutbin`, `mutde`, `mutinv`, `mutmove`, `mutswap`, `mutpp`, `mutate`
- **Multi-objective, constraints and migration (`multiobjective.rs`)**: `ndsortESS`, `ndsortTNS`, `ndsortDED`, `crowdis`, `cdist`, `mergecv`, `tcheby`, `pbi`, `crtidp`, `awGA`, `rwGA`, `refselect`, `refgselect`, `migrate`
- **Indicators (`indicator.rs`)**: `GD`, `IGD`, `Spacing`, `HV`

---

## 📄 Citation & License

If you use this project in academic work, please cite the original project:

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

Like the original, this project is licensed under the **[GNU Lesser General Public License v3.0 (LGPL-3.0)](LICENSE)**.

---

## 👤 Maintainer & Copyright

- **Maintainer of the Rust Core Edition**: yangyachao ([www.yangyachao.com](https://www.yangyachao.com))
- **Email**: [yangyachao@live.cn](mailto:yangyachao@live.cn)
- **Repository**: [https://github.com/yangyachao/geatpy](https://github.com/yangyachao/geatpy)

### Copyright Notice
- **Rust core re-implementation and distribution**: Copyright (C) 2026 yangyachao.
- **Original algorithm design and Python architecture**: Copyright (C) 2018-2021 Geatpy Development Team.
- Licensed under the **[LGPL-3.0 License](LICENSE)**.
