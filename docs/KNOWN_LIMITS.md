# Known Limits

- The sanitized replay proves orchestration behavior only. It provides no causal, out-of-sample,
  economic, or tradability evidence for either demo expression.
- No real WorldQuant request was made during development. Default live endpoint paths and response
  normalization were tested against intercepted HTTP only and remain unverified against an
  authorized current account.
- Live execution performs one poll per command. There is no scheduler, daemon, background polling,
  or notification service.
- The live adapter depends on a user-exported session-cookie file. It does not automate login or
  bypass human verification.
- Platform failures are persisted but do not automatically generate a repair. A user must
  explicitly start any next proposal or repair action.
- LLM services are optional library components, not an autonomous end-to-end loop. Their outputs
  require complete typed validation and remain subject to deterministic gates.
- One run freezes one dataset and one `region x universe x delay` cell. Cross-cell comparison and
  portfolio construction are outside the current package.
- SQLite and the synchronous orchestration model target a single local operator, not distributed
  or multi-tenant production workloads.
- Lexical retrieval uses approved paraphrased method cards and may miss semantic matches. Raw
  private corpus text is intentionally neither indexed nor persisted.
- MCP is replay-only and path-locked by the host process. It is not a remote control plane.
- There is no dashboard and no automatic submission API, CLI command, MCP tool, or adapter method.
