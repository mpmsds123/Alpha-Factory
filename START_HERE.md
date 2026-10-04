# Alpha 工厂：你每天要做什么

设计依据见设计文档第 2 版（claude.ai/artifact/UmRiCRnDTXLsPVVSH5u7RD）。

## 第一次使用（只做一次）

```bash
cd ~/Desktop/brain-alpha-pipeline
source .venv/bin/activate
pip install -e .                       # 安装 alpha-factory 命令
alpha-factory schedule install         # 每天 00:37 自动运行
```

还需要两个本机文件，都放在仓库外，不会进 git：

- `~/secrets/platform-brain.json`：BRAIN 账号，`{"email": "...", "password": "..."}`。你已经有了。
- `~/secrets/deepseek.json`：DeepSeek key，`{"api_key": "sk-..."}`。没有这个文件也能跑，
  只是不会调用 LLM（探测、扩展、修复照常进行，不会提假设和写表达式）。

## 每天

| 什么时候 | 做什么 |
|---|---|
| 收到"需要刷脸"通知 | `alpha-factory login --then-run`，刷完会从断点接着跑 |
| 想看进度 | `alpha-factory dashboard`（打开本地面板） |
| 有待批假设（每周一次即可） | `alpha-factory hypotheses list`，然后 `approve ID ...` 或 `reject ID ...` |
| 有待确认 alpha | `alpha-factory review list`，然后 `review approve ID`（真正提交）或 `review reject ID` |

**提交只会发生在你运行 `review approve` 的时候。** 程序自己永远不提交。

## 它每天自己做的事

1. 刷新你账号能用的地区、数据集、乘数和 pyramid 进度。
2. 按期望价值给每个"地区 × 数据集"打分；30% 预算用来探测没摸过的，其余投给有信号的。
3. 新数据集：字段分家族，每家派代表做一阶探测；50 次内没有 |Sharpe| ≥ 1 就止损。
4. 有信号的数据集：从灵感库（3,276 篇论文和论坛帖）检索相关内容，LLM 提假设；
   你批准后，LLM 按假设写表达式，再由另一次调用复核"表达式是否真的实现了假设"。
5. 回测结果自动分流：强负信号取反；过了扩展门槛的套分组、再套交易条件；
   没过的按失败原因修（每次只改一个变量，没过微调门槛不许调参数）。
6. 过了提交门槛的，挑每个数据集最好的两条做稳健性检验（rank、sign、窗口扰动、
   换中性化、换小股票池），保住一半以上 Sharpe 才算通过。
7. 平台检查 + 日 PnL 相关性去重 + 组合价值排序，选至多 4 条，LLM 写说明，等你确认。

## 文件在哪里

- 账本：`~/.alpha_factory/factory.db`（所有回测、决定、LLM 调用）
- 登录 cookie：`~/.alpha_factory/brain_cookies.json`（权限 600）
- 运行日志：`~/.alpha_factory/run.log`
- 面板：`reports/dashboard.html`（含真实表达式，已 gitignore）
- 所有阈值：`factory.yaml`，每个数字的出处写在 `src/alpha_orchestrator/factory/config.py`

## 不要做的事

- 不要把 `reports/`、账本、cookie 或任何含 BRAIN 接口的代码推到公开仓库（官方明令禁止）。
- 不要把 `auto_submit` 改成 true。
