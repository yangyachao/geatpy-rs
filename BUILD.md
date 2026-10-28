# Geatpy 打包与构建指南 (Build & Packaging Guide)

本项目核心算子已采用 **Rust (PyO3 + abi3-py38)** 完整重写，兼具极致计算性能与跨平台、跨 Python 版本的通用兼容性。

---

## 快速打包脚本索引

| 脚本文件 | 适用场景 | 常用命令 |
| :--- | :--- | :--- |
| [`scripts/build.sh`](file:///home/kaiquan/kaiquan/geatpy/scripts/build.sh) | Linux / macOS 本地一键打包与安装 | `./scripts/build.sh --test` |
| [`scripts/build.py`](file:///home/kaiquan/kaiquan/geatpy/scripts/build.py) | Windows / macOS / Linux 跨平台构建 | `python scripts/build.py --test` |
| [`scripts/build_wheels.sh`](file:///home/kaiquan/kaiquan/geatpy/scripts/build_wheels.sh) | 生产级发布包 (Wheel + sdist) 导出 | `./scripts/build_wheels.sh` |
| [`.github/workflows/release.yml`](file:///home/kaiquan/kaiquan/geatpy/.github/workflows/release.yml) | GitHub Actions 多平台自动化发布流水线 | 推送标签 `v*` 自动触发 |

---

## 一、本地开发与构建

### 1. 前置依赖
- **Python**: `>= 3.8` (支持 Python 3.8 ~ 3.15)
- **Rust**: 安装 `cargo` 与 `rustc` (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Maturin**: `pip install maturin>=1.5`

### 2. 常用开发命令

#### 模式 A：使用 `build.sh` (Linux / macOS)
```bash
# 1. 编译并打包 Wheel 到 dist/ 目录
./scripts/build.sh

# 2. 编译并直接安装到当前 Python 环境
./scripts/build.sh --install

# 3. 编译、安装并运行全部单元测试
./scripts/build.sh --test

# 4. 同时生成源码包 (sdist .tar.gz)
./scripts/build.sh --sdist
```

#### 模式 B：使用 `build.py` (Windows / Linux / macOS 通用)
```bash
# 1. 编译并打包 Wheel 到 dist/ 目录
python scripts/build.py

# 2. 编译、安装并运行单元测试
python scripts/build.py --test

# 3. 同时生成源码包 (sdist .tar.gz)
python scripts/build.py --sdist
```

#### 模式 C：标准 pip 安装
```bash
# 开启 editable 模式进行代码开发
pip install -e .

# 或者直接从源码安装
pip install .
```

---

## 二、生产级发布打包 (Wheel & sdist)

运行发布打包脚本：
```bash
./scripts/build_wheels.sh
```

构建完成后产物位于 `dist/` 目录下：
- `dist/geatpy_rs-2.7.5-cp38-abi3-<platform>_<arch>.whl`：针对该操作系统的二进制 Wheel。
- `dist/geatpy_rs-2.7.5.tar.gz`：通用的源码分发包 (sdist)。

> **特性说明**：由于开启了 Python **abi3-py38** 特性，编译出的单个二进制 Wheel 可直接在 **Python 3.8、3.9、3.10、3.11、3.12、3.13、3.14、3.15** 下直接加载运行，无需针对每个 Python 小版本分别编译。
>
> free-threaded（无 GIL）的 CPython 没有稳定 ABI：用 3.14t 解释器构建时（如 `python3.14t scripts/build.py`），PyO3 会给出
> “abi3 does not yet support CPython 3.14t” 的警告并改为构建该版本专用的 `cp314t` Wheel，属正常现象。

发布到 PyPI：
```bash
pip install twine
twine upload dist/*
```

---

## 三、GitHub Actions CI/CD 多平台自动化流水线

项目配置了完整的自动化构建流程：
1. **持续集成 ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))**：
   - Rust fmt / clippy（含测试代码）与单元测试（Ubuntu x86_64、macOS arm64），并在 Ubuntu、macOS、Windows 上测试 Python 3.8 至 3.15。
   - 平时推送不运行，只在发布时由发布流水线调用；也可以在 Actions 页面手动运行（Run workflow）。
2. **多平台发布流水线 ([`.github/workflows/release.yml`](.github/workflows/release.yml))**：
   - 当向仓库推送标签（如 `git push origin v2.7.5`）时自动启动，先运行上面的 CI，全部通过后才发布。
   - 自动化跨平台编译构建：
     - `manylinux` x86_64
     - `manylinux` aarch64 (ARM64)
     - macOS universal2 (兼容 Intel 与 Apple Silicon M1/M2/M3/M4)
     - Windows x86_64
     - 源码包 `sdist`
   - 自动生成 GitHub Release 附件并发布至 PyPI。

---

## 与官方 2.7.0 内核的一致性测试

```bash
# 快速部分（确定性算子逐值比对 + 随机算子统计比对），CI 中默认运行
pytest tests/test_core_parity.py

# 39 个算法模板的解质量对比（较慢）
GEATPY_PARITY_TEMPLATES=1 pytest tests/test_core_parity.py -k template

# 重新生成官方内核的基准数据（需要 Docker；在 linux/amd64 + Python 3.6 中运行官方二进制）
./scripts/parity/run_original_core.sh --reps 5
# 只更新算子部分、保留模板结果
./scripts/parity/run_original_core.sh --skip-templates --merge
```

用例定义在 `tests/parity_cases.py`，原版与 Rust 两侧运行同一份代码。官方二进制中与其文档不符的缺陷登记在
`KNOWN_ORIGINAL_DEFECTS`，对应算子按文档语义实现并由独立测试覆盖。

## macOS：`mis-aligned LINKEDIT string pool`

部分新版 Apple 链接器（如 Xcode 27 的 ld-27037）生成的动态库字符串表只有 4 字节对齐，导入时 dyld 会报
`mis-aligned LINKEDIT string pool`。`scripts/build.py --install` 会自动检测并修复；手动修复：

```bash
python scripts/macos_fix_linkedit.py "$(python -c 'import importlib.util as u;print(u.find_spec("_geatpy_core").submodule_search_locations[0])')"/_geatpy_core.abi3.so
```

GitHub Actions 上构建的发布包不受影响。
