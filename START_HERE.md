# Alpha 工厂：你每天要做什么

依据：官方《零基础学量化》AI 版第 1–4 课、官方 API 课、cnhkmcp 的数据探测方法、论坛作业区。

## 第一次使用（只做一次）

```bash
cd ~/Desktop/brain-alpha-pipeline
source .venv/bin/activate
pip install -e .                       # 安装 alpha-factory 命令
alpha-factory ideas import ~/Downloads/WQ官方培训作业帖全文.md   # 作业区模版进灵感库（一次）
alpha-factory schedule install         # 每天 00:37 自动运行
```

还需要两个本机文件，都放在仓库外，不会进 git：

- `~/secrets/platform-brain.json`：BRAIN 账号，`{"email": "...", "password": "..."}`。你已经有了。
- `~/secrets/deepseek.json`：DeepSeek key，`{"api_key": "sk-..."}`。出点子和读结果都靠它，
  没有这个文件程序不运行。

## 每天

| 什么时候 | 做什么 |
|---|---|
| 收到"需要刷脸"通知 | `alpha-factory login --then-run`，刷完会从断点接着跑 |
| 想看进度 | `alpha-factory dashboard`（打开本地面板） |
| 想先过目 LLM 的点子（可选） | `factory.yaml` 里 `hypotheses_need_approval: true`，之后 `alpha-factory hypotheses list` / `approve ID` |
| 有待确认 alpha | `alpha-factory review list`，然后 `review approve ID`（真正提交）或 `review reject ID` |

**提交只会发生在你运行 `review approve` 的时候。** 程序自己永远不提交。

## 它每天自己做的事

1. **选数据集（回测之前，只选一次）**：按资料排序，不花回测。依据是平台 Research Papers
   是否点名这个数据集、金字塔乘数和你还缺的格子、已有 alpha 数（拥挤度）、你组合里这个类别
   的占比、地区成本和 Sharpe 门槛。同时做 2 个，停掉一个就按顺序开下一个。
2. **读懂数据（约 60 次回测）**：字段按近似重复分家族，最大的 10 个家族各派一个代表，
   跑官方"6 种方法"（中性化关、decay 0，只看多空股票数）：覆盖率、非零、更新周期、
   取值范围。结果决定每个字段进模版时怎么预处理（补数据多少天、vector 怎么压、要不要去极值）。
3. **LLM 出点子**：读字段说明和数据画像，加上点名这个数据集的论文、作业区和论坛里最相近的
   模版，提几个点子。每个点子写明机制、方向、模版、适用字段、窗口、和窗口相配的 decay。
4. **展开回测**：每个点子按模版展开到它的字段和窗口上，打乱后取至多 60 条。
5. **LLM 读结果定下一步**：一轮跑完，LLM 看每个点子的 Sharpe 分布、各字段最好成绩、
   失败原因，逐个判断：有矿就在附近深挖（写变体 + 分组比较）、方向反了就取反、
   被某个检查卡住就修、没信号就放弃；再决定这个数据集是继续、要新点子、还是放弃。
6. **停止规则（程序强制，LLM 改不了）**：连续 2 轮没有任何 |Sharpe| ≥ 1、数据集用满
   1,000 次回测、或交满 8 个，就停，停的原因写在面板上。
7. **提交前**：过门槛的挑最好的做稳健性检验（rank、sign、窗口扰动、换中性化、换小股票池），
   保住一半以上 Sharpe 才算过；再过平台检查 + 日收益相关性去重，选至多 4 条，LLM 写说明，等你确认。

预算：并发 10 × 12 小时 × 每小时约 20 次 ≈ 每天 2,400 次（`factory.yaml` 里改）。

## 文件在哪里

- 账本：`~/.alpha_factory/factory.db`（所有回测、决定、LLM 调用）
- 登录 cookie：`~/.alpha_factory/brain_cookies.json`（权限 600）
- 运行日志：`~/.alpha_factory/run.log`
- 面板：`reports/dashboard.html`（含真实表达式，已 gitignore）
- 所有阈值：`factory.yaml`，每个数字的出处写在 `src/alpha_orchestrator/factory/config.py`

## 不要做的事

- 不要把 `reports/`、账本、cookie 或任何含 BRAIN 接口的代码推到公开仓库（官方明令禁止）。
- 不要把 `auto_submit` 改成 true。
