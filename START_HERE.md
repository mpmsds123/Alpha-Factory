# 你只需要做什么

平时只运行这一条：

```bash
tools/daily_run.sh
```

它会自己完成：读取上次未完成的 run → 用已批准 Method Cards 生成候选 → 去重与结构检查 →
按账户上限 10 自适应调度模拟 → 持续回收 pending → 立即检查幸存候选 → 做相关性门禁 → 生成报告。

你只会在两种情况下被叫回来：

1. BRAIN 会话过期或要求真人验证：运行 `python3 tools/取会话.py`，然后重新执行
   `tools/daily_run.sh`；它会自动续跑保存的 run，不会重新建一批。
2. 出现 `READY_FOR_HUMAN_REVIEW`：系统已经完成研究与平台检查，需要你决定是否手工提交。

系统永远不会自动提交 Alpha。数据库 `alpha_lab.db` 是唯一账本；`.runtime/active_run_id`
只是断点指针，完成后会自动清除。

## 当前证据边界

- 代码、回放和被拦截的 HTTP 测试通过，不等于真实平台成功。
- 只有 BRAIN 返回的 live checks/correlation 才能把候选送到人工审核。
- `PASSED_LOCAL_SCREEN`/`READY_FOR_PLATFORM_CHECK` 不是可提交结论。
