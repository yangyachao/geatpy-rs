# geatpy-rs 协作约定

geatpy 2.7.0 的 Python 层原样保留，闭源 Cython 内核 `_core` 用 Rust 重写（`crates/geatpy_core`，PyO3 abi3，
单一 `cp38-abi3` 轮子）。目标是**与官方 2.7.0 二进制行为一致**：同样的输入得到同样的输出、同样的返回类型、
同样的报错；随机算子则分布一致。PyPI 包名 `geatpy-rs`，导入名仍是 `geatpy`。

## 目录

- `crates/geatpy_core/src/`：Rust 内核。`lib.rs` 注册全部导出函数；`encoding.rs`（crt*、bs2*、ri2bs、boundfix）、
  `selection.rs`（选择、ranking/scaling/powing/indexing、otos、ecs）、`recombination.rs`（rec*、xov*、recombin）、`mutation.rs`（mut*、mutate）、
  `multiobjective.rs`（ndsort*、crowdis、cdist、mergecv、tcheby、pbi、awGA、rwGA、crtidp、refselect、refgselect、
  migrate）、`indicator.rs`（GD、IGD、Spacing、HV）、`utils.rs`（参数解析、FixType 修复、int32 输出）。
- `geatpy/`：原版 Python 层（算法模板、Problem、Population 等），一般不改；只修原版缺陷，或做不改变行为的性能优化
  （如 `Population`/`PsyPopulation` 的花式索引与合并直接接管新数组、不再二次复制，`stat()` 只取最优个体）。
  改后用同一种子对比各模板的结果哈希，并保证新种群与原种群不共享内存（`tests/test_population_copy.py`）。
- `tests/`：`test_core_parity.py` 与官方二进制的黑盒差分（期望值在 `tests/fixtures/core_parity_orig_2_7_0.json`）；
  `test_core_parity_review.py` 逆向复核得到的回归用例。
- `reverse/`（已 gitignore，只在本机）：官方 `_core` 的反编译结果，仅供审查。
  - `reverse/compact/core/<模块>.c`：该模块的纯 C 核心函数（算法主体，优先读这个）；
  - `reverse/compact/outline/<模块>.txt`：Cython Python 层（`__pyx_pf_`）按顺序出现的名字、常量与运算，
    用于确认参数默认值、校验、返回类型；
  - `reverse/compact/with_pf/<模块>.c`：纯 C 核心函数加 Cython `__pyx_pf_` 包装函数的反编译正文（outline 不够时查参数处理细节）；
  - `reverse/decompiled/<模块>.c`：完整反编译（很大，只在上面几份不够时查具体函数）；
  - `reverse/REVIEW_FINDINGS.md`：上一轮审查报告。

## 已知且有意保留的差异

以下差异已确认，审查时不必再报（除非发现记录本身有误）：

- 原版自身缺陷、Rust 按文档实现：`ndsortDED` 分层错误、`mergecv` 跨行累加、`tcheby`/`pbi` 在 `idealPoint=None`
  时返回理想点、`refselect` 候选矩阵行序错位及无需选择时返回 arange、`migrate` 的 `Replacement=2` 淘汰最优个体、
  `mutgau`/`mutuni` 的 `Middle=True` 以 `(ub-lb)/2` 为中心、`crtip`/`crtri`（含 `crtpc('RI')` 与手工 FieldDR 的
  `initChrom`）整数列边界为小数时越界（区间内没有整数时 Rust 取 ceil(lb)）。
- `mutde` 的 F 列表中的 None：原版取另一个因子（`[0.5, None]` 等于 `[0.5, 0.5]`），`[None, None]` 还报“F 必须大于 0”，
  与文档示例“F2 在变异过程中随机生成”和 currentToRand 模板的注释矛盾；Rust 按文档为每个个体随机生成。因此
  `soea_DE_currentToRand_1_templet` 在 Rastrigin 上比 2.7.0 收敛慢（50×200 中位最优值约 +6，30 维 50×1000 约 93 → 156），
  属预期。`Mask_N` 为恰好 N 行的矩阵时，原版不把个体自身排除在供体之外（与“r0 不等于 i”矛盾），Rust 与整数
  `Mask_N=N` 一样排除。
- 原版缺陷（上游 issue 确认）：`mselecting` 只给一个都没选中的种群补选，文档保证的每个种群至少 ceil(规模×MSel)
  个做不到（geatpy-dev/geatpy#330）；Rust 给不足的种群补足下限，其余情形结果与原版相同，总数因此可能多于 NSel。
- 原版缺陷（第三轮差分确认）：`migrate` 的 `Replacement=1` 总淘汰最后一个个体、`Structure=0` 迁入来源含自身；
  `pbi` 个体在理想点时为 NaN；`cdist` 对非连续数组读错；长度为 1 或一维输入时段错误；
  HV 内部 float32；`sus` 二进制要求至少 3 个参数。
- 原版缺陷（2026-10-20 审查确认），Rust 报干净的异常或给出合理结果，不照搬：
  - 越界读或读未初始化内存：`mutde` 二维 Xr 行数不足或列数不符、Mask_N 列数不符（Rust 报 RuntimeError）；
    `refgselect` 的 Gamma 短于参考点数（ValueError）、CV 行数少于 ObjV（RuntimeError，多于 ObjV 时原版只用前 n 行，
    Rust 同样报错，文本为原版 ndsort* 的 “CV and ObjV disagree”）；GD 的 PF 为空、IGD 的 ObjV 为空（Rust 为 inf）；
    HV 的 PF 为空（Rust 为 0.0）；三维 ObjV/参考点在 cdist、ndsortESS/DED、GD/IGD/Spacing、mergecv 中被当作二维
    读取（Rust 报 ValueError 'Buffer has wrong number of dimensions'）。
  - 崩溃：`mutde` 空种群且需随机抽供体时除零（Rust 报 RuntimeError）；`refselect` 的 levels 为标量时可能段错误；
    分配失败或尺寸溢出时 Rust 与原版一样报 MemoryError / ValueError（Rust 在按用户输入定尺寸的分配前先检查）。
  - `refselect`/`refgselect` 的参考点集为空时原版给出任意选择，Rust 报 ValueError。
  - `refgselect` 就地把调用者 CV 的负值置 0（RVEA 系模板返回种群的 CV 因此没有负值），Rust 不修改输入；ObjV 为空且
    不传 CV 时原版报 numpy 的 TypeError，Rust 返回空选择和 Gamma。
  - GD/IGD 的 `Parallel=True` 有数据竞争，偶尔返回错误的值；Rust 始终是串行结果。整数或 float32 输入在原版 GD/IGD/
    Spacing 中得到垃圾值，Rust 转为 float64。
  - NaN：GD/IGD 跳过含 NaN 的候选点（原版只有 NaN 恰在一行末尾时才得到 NaN），被平均的点到所有候选的距离都是 NaN
    时结果为 NaN；HV 排除含 NaN 的点（原版是否得到 NaN 取决于位置和 M），PF 某列全是 NaN 时为 NaN。
- 原版缺陷（2026-10-28 确认）：`recint`/`reclin` 的 Alpha 为整数、float32 或 bool 数组时，原版按 float64 读取其内存
  （值是垃圾，或随机报“Some elements of Alpha are illegal”）；Rust 转为 float64，bool 数组报该错误。
  `bs2int` 的 FieldD 码长为 0 时除以 2^0 - 1 = 0：x86_64 原版整数除零崩溃，arm64 原版得到下界，Rust 取下界。
- 原版缺陷（2026-10-28 全量审查确认），Rust 不崩溃、给出合理结果：
  - 空种群：`rws`/`sus` 返回未初始化的下标（常为 0），Rust 返回全 0；`tour`/`dup` 原版崩溃，Rust 返回空数组
    （`urs` 与原版一样报 numpy 的 `high <= 0`，`mselecting` 空列表报 `need at least one array to concatenate`）。
  - `migrate` 用到适应度（Select=1 或 Replacement=2）时，FitnVs 个数少于种群数或某个 FitnV 行数不足，原版读越界；
    Rust 报 RuntimeError。不用适应度时两边都不读 FitnVs。
  - 非线性 `ranking`（RM=1）在 N、SP 较大时 x^(N-1) 溢出，原版得 NaN；Rust 改用 1/x 的幂计算，结果正确
    （最大值 SP、总和 N）；不溢出时公式与结果不变。
  - FieldDR 上下界反序或含 NaN：截断修复与原版一样先比下界、再比上界（NaN 边界不越过）；`crtrp`/`crtri` 的实数变量
    与原版一样取 lb + u·|ub - lb|（区间为 inf/NaN 时得 inf/NaN）；`crtip`/`crtri` 的整数变量反序时 Rust 取 ceil(lb)
    （原版是 lb 以上的随机整数）。
- Python 层修正的原版缺陷（2026-10-28）：切片 `pop[a:b:c]` 的个体数按实际选中的行计算（原版 `pop[:]` 报 TypeError、
  `pop[:-1]` 报错、越界的 stop 记成错误的 sizes）；`__setitem__` 在 Phen 为 None 时不再报错；`PsyPopulation.initChrom`
  重复调用不再累加 Linds；`PsyPopulation.shuffle` 在 Encodings 为 None 时不再报错；`PsyPopulation` 逐个复制 Fields，
  派生种群不再共享译码矩阵。
- `refselect` 极点矩阵奇异（多为同一个体同时是几个目标的极点）时，原版 numpy LU 因舍入得到 det≈1e-19≠0，
  解出无意义（甚至为负）的截距；Rust 用除法消元得到精确零主元，退回列最大值（随机 3 目标用例约 1.5%，
  8–12 目标更常见）。其余步骤按原版 arm64 轮子的舍入实现：LU 主元按 OpenBLAS `idamax`（含 NaN 时的行为）、
  余弦为先归一化再 FMA 点积、`1-cos²` 最小者关联、`d` 为负时 NaN；非奇异时求解的舍入仍可能不同（约 0.05%）。
- `refgselect` 同样按原版舍入实现，与官方逐位一致（含 Gamma）：余弦距离 `1-cos` 最小者关联、Gamma 为 `acos`
  （不 clamp，值等于 1 的余弦先置 0 并发出原版的 RuntimeWarning）、APD 用 `acos(1-d)`；CV 负值置 0 后按 numpy 求和，
  NaN 保留并按 arm64 原版视为不可行（x86 反编译的 `1e-12 < CV` 写法看不出这一点）。
- `rws` 先把 FitnV 平移到非负；`crowdis` 接受 `levels=None`；HV 在 M=4 时用精确算法（原版蒙特卡洛）；M ≥ 5 与原版一样用 10 万个样本的蒙特卡洛（估计值的方差与原版相同）。
- 有意更宽松（原版报错）：`maxormins` 接受 list；'P' 编码的 mutinv/mutmove/mutswap 不强制 FieldDR；一维输入；
  整数参数（Half_N、Nind、Mask_N 等）接受 numpy 整数（原版只认 Python int）；一维参数（refselect 的 levels、
  refgselect 的 Gamma 等）接受 list、整数数组和 n×1/1×n 数组；`mutde` 的 F 接受 tuple 和长度不为 2 的列表
  （单元素当标量，空列表当 None；原版报错或段错误）；经 `geatpy.core` 包装层调用时关键字参数不区分大小写和下划线
  （如 `neednum=`）。
- 有意更严格或不同：`refselect` 的 NUM 为负时报 OverflowError（原版按 0 处理）；`ndsort*` 的 needNum/needLevel 超出
  C int 范围（含 inf）时按全部分级处理（原版 ESS/TNS 报 OverflowError）。
- GeneID 与原版一样截断为 int32；NaN 按 arm64 上 numpy 的转换取 0（x86_64 原版得到 INT_MIN，排序位置不同）。
- 整数输出一律 int32（与原版一致）；例外是原版用 numpy 实现的 `crtbp`、`urs`、`ecs`、Nsel = N 的 `otos`（int64），
  以及长度 < 2 的排列染色体原样返回。
- 并行参数 `Parallel` 目前不生效（为与原版随机序列语义对齐移除了 rayon）。

## 验证

| 改动范围 | 验证要求 |
| --- | --- |
| Rust 内核 | `maturin develop --release` 后 `python -m pytest tests -q`；`cargo fmt --check`；`cargo clippy --all-targets -- -D warnings`；`cargo test --lib` |
| 有意改变输出 | 在 Docker 中用官方二进制重新生成黄金文件（`scripts/parity/run_original_core.sh`），提交说明写明原因 |
| 改了算子行为或校验 | 用 `scripts/parity/fuzz/` 对官方二进制做随机差分（见其 README，官方 macOS arm64 轮子可在本机原生运行） |
| 改了随机算子或性能路径 | 另跑 `scripts/parity/classic.py`（36 个经典用例，两边各 30 次，`cmp` 中 \|z\| > 3 需复查；`rastrigin_multi_SEGA` 因 migrate 缺陷修正而显著更好，属预期；F 列表含 None 的 currentToRand 模板按文档随机 F2，比原版差，也属预期） |
| 改了参数校验、报错或边界输入 | 合法输入之外的情形（空、反向/非连续、三维、超大尺寸、NaN、错误类型）对照官方二进制补进 `tests/test_core_parity_review4.py` 一类的回归测试：fuzz 只覆盖合法输入 |
| 仅文档 | 检查内容与引用路径 |

- 浮点排序用全序比较（`crate::utils::cmp_nan_last`，即 numpy 的 NaN 置后顺序），不要用 `partial_cmp(..).unwrap()` 或
  `unwrap_or(Equal)`：后者遇到 NaN 不是全序，Rust 1.81 起排序会 panic。目标值含 NaN 时 `ndsort*` 调用 numpy 的
  `np.unique`/`np.argsort`，与原版一样由 numpy 的快速排序决定相等行的先后。
- 随机数一律用 `crate::utils::rng()`（每次调用从 numpy 全局 RandomState 取种子，`np.random.seed` 才能复现），不要用 `thread_rng`。
- 性能优化只改实现、不改算法：每个基因/个体的概率与原版相同（例如 `mutbin` 用几何分布跳跃抽取变异位），改后用 fuzz 与
  `scripts/parity/classic.py` 确认分布与解质量不变。
- 提交信息用英文 Conventional Commits；README 与设计文档用中文；Rust 注释用英文。
- `README.en.md` 是 README 的英文版，改 README 时同步更新（版本号、数字、缺陷清单两边一致）。
- `reverse/` 下的反编译内容不得提交或发布。
