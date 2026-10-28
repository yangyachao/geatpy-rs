#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geatpy 跨平台打包与构建脚本 (Windows / macOS / Linux 通用)

用法:
    python scripts/build.py                # 编译并生成 wheel 至 dist/
    python scripts/build.py --install      # 编译、打包并安装到当前 Python 环境
    python scripts/build.py --test         # 编译、安装并运行 pytest 单元测试
    python scripts/build.py --sdist        # 额外生成源码包 (.tar.gz)
    python scripts/build.py --release      # 发布优化模式 (默认开启)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stream handling if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def run_cmd(cmd, check=True):
    print(f"\033[34m[CMD]\033[0m {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        print(f"\033[31m[ERROR] Command failed with exit code {res.returncode}\033[0m")
        sys.exit(res.returncode)
    return res.returncode


def ensure_toolchain():
    # Check cargo
    cargo_bin = shutil.which("cargo")
    if not cargo_bin:
        cargo_home = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
        if cargo_home.exists():
            os.environ["PATH"] = str(cargo_home.parent) + os.pathsep + os.environ.get("PATH", "")
            cargo_bin = str(cargo_home)
        else:
            print("\033[31m[ERROR] Rust toolchain (cargo) not found. Please install Rust from https://rustup.rs/\033[0m")
            sys.exit(1)

    print(f"\033[32m[INFO] Rust toolchain:\033[0m {cargo_bin}")

    # Check maturin
    try:
        import maturin  # noqa: F401
    except ImportError:
        print("\033[33m[INFO] Installing maturin build tool...\033[0m")
        run_cmd([sys.executable, "-m", "pip", "install", "maturin>=1.5"])


def repair_macos_linkedit():
    """On macOS, re-align the installed extension if dyld rejects it (local linker bug, see
    scripts/macos_fix_linkedit.py). No-op everywhere else."""
    if platform.system() != "Darwin":
        return
    probe = subprocess.run([sys.executable, "-c", "import _geatpy_core"], capture_output=True, text=True)
    if probe.returncode == 0 or "mis-aligned LINKEDIT" not in probe.stderr:
        return
    locate = subprocess.run([sys.executable, "-c", "import importlib.util as u, os; "
                             "print(os.path.dirname(u.find_spec('_geatpy_core').origin))"],
                            capture_output=True, text=True, check=True)
    for so in Path(locate.stdout.strip()).glob("_geatpy_core*.so"):
        print(f"[INFO] Re-aligning LINKEDIT string table of {so.name} (macOS linker workaround)")
        run_cmd([sys.executable, str(REPO_ROOT / "scripts" / "macos_fix_linkedit.py"), str(so)])


def main():
    parser = argparse.ArgumentParser(description="Geatpy cross-platform automated build tool")
    parser.add_argument("--install", "-i", action="store_true", help="Install built wheel to current environment")
    parser.add_argument("--test", "-t", action="store_true", help="Build, install and run pytest test suite")
    parser.add_argument("--sdist", "-s", action="store_true", help="Build source distribution (.tar.gz)")
    parser.add_argument("--debug", "-d", action="store_true", help="Build in debug mode (default is release)")
    parser.add_argument("--target", help="Specify compilation target architecture")
    args = parser.parse_args()

    print("=" * 60)
    print("      Geatpy (Rust Core) Cross-Platform Build Tool")
    print(f"      Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"      Working Dir: {REPO_ROOT}")
    print("=" * 60)

    ensure_toolchain()

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "-o",
        str(DIST_DIR),
        # this interpreter: abi3 wheel normally, a cp314t wheel on free-threaded CPython
        "-i",
        sys.executable,
    ]

    if not args.debug:
        build_cmd.append("--release")

    if args.sdist:
        build_cmd.append("--sdist")

    if args.target:
        build_cmd.extend(["--target", args.target])

    print("\n\033[32m[1/3] Building Geatpy Wheel...\033[0m")
    run_cmd(build_cmd)

    # Find latest wheel
    wheels = sorted(DIST_DIR.glob("geatpy*.whl"), key=os.path.getmtime, reverse=True)
    if not wheels:
        print("\033[31m[ERROR] No wheel found in dist directory\033[0m")
        sys.exit(1)

    latest_wheel = wheels[0]
    print(f"\n\033[32m[2/3] Build succeeded! Wheel:\033[0m {latest_wheel.name} ({latest_wheel.stat().st_size / 1024:.1f} KB)")

    # Install
    if args.install or args.test:
        print(f"\n\033[32m[3/3] Installing wheel to current environment...\033[0m")
        in_venv = sys.prefix != sys.base_prefix
        uv_bin = shutil.which("uv")
        installed = False
        if uv_bin:
            res = run_cmd([uv_bin, "pip", "install", "--reinstall", "--no-deps", "--break-system-packages", str(latest_wheel), "--python", sys.executable], check=False)
            if res == 0:
                installed = True
        if not installed:
            pip_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--break-system-packages", str(latest_wheel)]
            if not in_venv:
                pip_cmd.append("--user")
            run_cmd(pip_cmd)
        repair_macos_linkedit()
        print("\033[32m[DONE] Installation completed successfully!\033[0m")

    # Test
    if args.test:
        print("\n\033[32m[TEST] Running pytest test suite...\033[0m")
        run_cmd([sys.executable, "-m", "pytest", str(REPO_ROOT / "tests"), "-v"])
        print("\033[32m[DONE] All tests passed!\033[0m")


if __name__ == "__main__":
    main()
