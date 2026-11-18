# quant-falsifier

[![CI](https://github.com/Zoey-nju/quant-falsifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoey-nju/quant-falsifier/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/quant-falsifier.svg)](https://pypi.org/project/quant-falsifier/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**量化研究的证伪工具。** 一个依赖极少的 Python 库，只问回测一个不太好回答的问题：
*在它成立之前，你一共试了多少个版本？*

> English version: [README.md](README.md)

---

## 动机

夏普比率不是一个结果，而是一次未披露搜索的最大值。试 200 组动量参数，最好那组的曲线
一定好看——即使 200 组全都是噪声，200 个零技能夏普比率的最大值期望也严格大于零。同样的
机制藏在交叉验证里：标签区间重叠会把测试集信息漏进训练集，"样本外"悄悄变成了"样本内"。

`quant-falsifier` 就是夹在"我的 notebook 说夏普 2.1"和"我能为这个结论负责"之间的一层薄薄
的工具。它不生成策略、不下单，只做三件事：记录搜索、为搜索次数修正统计量、检查评估协议
是否存在泄露，最后输出一份写清自身局限的报告。

## 功能

| 模块 | 回答的问题 |
| --- | --- |
| `TrialLedger` | 到底试过什么、顺序如何、记录有没有被改过（只追加 JSONL + SHA-256 哈希链） |
| `deflated_sharpe_ratio` | 赢家能否经受住它背后试验次数的检验（Bailey & López de Prado） |
| `reality_check` / `spa_test` / `romano_wolf` | 最好的策略是否强于运气，哪些策略能通过族错误率校正（平稳自举） |
| `PurgedKFold` / `audit_splits` | 交叉验证协议是否泄露；可审计任何库产出的划分 |
| `check_point_in_time` | 是否有决策用到了当时尚未公布的数据 |
| `label_overlap` / `signal_delay_curve` / `permutation_pvalue` | 样本的有效独立量、延迟一根 bar 后优势是否还在、信号与收益的配对是否有意义 |
| `AuditReport` | Markdown 或自包含 HTML 报告，逐项判决并附局限说明 |

## 安装

```bash
pip install quant-falsifier
```

Python 3.10+，运行时只依赖 NumPy。任何接受数组的地方都可以直接传 pandas 对象，但不强制。

## 快速开始

### 1. 记录试验，再为赢家"通缩"

```python
import numpy as np
from quant_falsifier import TrialLedger, deflated_sharpe_ratio, per_period_sharpes

rng = np.random.default_rng(0)
trial_returns = rng.normal(0.0, 0.01, size=(900, 30))  # 30 个变体，900 根 bar
sharpes = per_period_sharpes(trial_returns)  # 单期口径，不要年化

ledger = TrialLedger("trials.jsonl")
for i, sr in enumerate(sharpes):
    ledger.append(
        f"variant_{i:02d}", params={"variant": i}, metrics={"sharpe_per_period": float(sr)}
    )

best = int(np.argmax(sharpes))
result = deflated_sharpe_ratio(trial_returns[:, best], ledger.sharpes())
print(f"观测 {result.observed_sr:.2f}，门槛 {result.expected_max_sr:.2f}，DSR {result.dsr:.3f}")
```

`DSR < 0.95` 表示：这个结果与"30 个毫无价值的策略里最好的那个"无法区分。这正是重点——
它是一道你可以输掉的检验。

### 2. 审计已有的交叉验证划分

```python
import numpy as np
from quant_falsifier import PurgedKFold, audit_splits

times = np.arange(1200, dtype=float)
label_ends = times + 5  # 5 根 bar 的前瞻收益，标签互相重叠

naive = audit_splits(
    [(np.setdiff1d(np.arange(1200), test), test) for test in np.array_split(np.arange(1200), 5)],
    times,
    label_end_times=label_ends,
)
print(naive.is_clean, naive.n_label_overlaps)  # False 20

purged = audit_splits(
    PurgedKFold(5, label_end_times=label_ends).split(times), times, label_end_times=label_ends
)
print(purged.is_clean)  # True
```

### 3. 生成可以贴进 PR 的审计报告

```bash
quant-falsifier demo --out demo_out
```

```
Honest audit: overall verdict WARN
  PASS  Trial ledger
  WARN  Deflated Sharpe ratio
  PASS  Reality check (data snooping)
  PASS  Superior predictive ability
  PASS  Romano-Wolf stepdown
  PASS  Cross-validation leakage (purged K-fold)
  PASS  Point-in-time availability (clean signal)
  WARN  Overlapping labels
  PASS  Signal delay sensitivity
  PASS  Permutation test
Negative controls: overall verdict FAIL
  FAIL  Cross-validation leakage (naive K-fold)
  FAIL  Point-in-time availability (leaky signal)
```

demo 全部使用种子生成的合成数据，不包含任何行情数据。它会写出两份报告：
`demo_out/report.{md,html}` 是对这次参数扫描的诚实审计；
`demo_out/negative_controls.{md,html}` 把同一份数据喂给两个**故意写错**的评估协议，
用来证明每个检查确实会报警。诚实报告落在 `WARN`，因为最优策略能通过单策略显著性检验
（未校正 PSR ≈ 0.9999），但通不过 30 次试验的通缩校正（DSR ≈ 0.9246）——这个落差
正是这个库存在的理由。哈希链账本写在 `demo_out/trials.jsonl`。

### 4. 在 CI 中校验账本

```bash
quant-falsifier ledger verify trials.jsonl   # 哈希链断裂时退出码为 1
```

## 设计原则

1. **不做静默清洗。** `NaN`、无穷、零方差、空输入一律报错。会替输入"擦屁股"的校验器，
   没有资格评判输入。
2. **单位显式。** 所有修正公式都在单期口径下计算，年化只发生在报告层。
3. **零假设显式。** 每个 p 值都写明对应哪个零假设，并采用 `(1 + 超越次数) / (B + 1)`，
   永远不会出现 p = 0。
4. **范围刻意小。** 四个模块、一个依赖、没有插件系统。这个项目有意思的地方，在于它
   拒绝宣称什么。

## 它不能做什么

- 它无法检测所有类型的未来函数。这些检查覆盖结构性泄露、时点可得性、标签重叠和延迟
  敏感性；如果一个信号建立在被悄然修订过的数据上，它仍然会通过。
- 它不能证明账本是完整的，只能证明账本没有被改写。只有严格的记录习惯能让试验次数为真。
- 它不建模交易成本、借券成本、容量与市场冲击。
- 它不是投资建议；一份通过的报告也不是预测。

## 路线图

- [ ] `v0.2`：夏普比率与回撤的块自举置信区间
- [ ] `v0.2`：组合清洗交叉验证（CPCV）与路径级过拟合概率
- [ ] `v0.3`：可选的 pandas/polars 适配层，保留索引元数据
- [ ] `v0.3`：报告 JSON Schema，便于 CI 做机器可读的门控

路线图记录在 [Issues](https://github.com/Zoey-nju/quant-falsifier/issues)，不构成承诺。

## 开发

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
ruff check . && ruff format --check .
mypy
```

提交规范见 [CONTRIBUTING.md](CONTRIBUTING.md)，各项检查的细节与边界见
[docs/limitations.md](docs/limitations.md)。

## 参考文献

- Bailey, D. H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. Journal of Risk 15(2).
- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. Journal of Portfolio Management 40(5).
- White, H. (2000). *A Reality Check for Data Snooping*. Econometrica 68(5).
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. JBES 23(4).
- Romano, J. P. & Wolf, M. (2005). *Stepwise Multiple Testing as Formalized Data Snooping*. Econometrica 73(4).
- Politis, D. N. & Romano, J. P. (1994). *The Stationary Bootstrap*. JASA 89(428).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7（purged cross-validation）。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
