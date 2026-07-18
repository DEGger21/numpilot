# ADR-001:策略模式实现可替换迁移引擎

- 状态:已接受，2026-07-18 修订
- 日期:2026-07-02
- 相关:[ADR-003](ADR-003-contract-first-agents.md)、[architecture.md](../architecture.md) C4-L3

## 背景(Context)

迁移的本质是把 `np.<op>` 改写为 `ap.<op>`。有两条可行路径:确定性的 **AST 改写**(离线、可复现、基于覆盖集泛化)与 **LLM 驱动改写**(更灵活但需联网、有不确定性)。二者各有场景:课程 demo 与 CI 需要零依赖可复现;未来参赛可能引入 LLM。迁移智能体不应与某一种具体实现耦合。

## 决策(Decision)

引入策略接口 `MigrationEngine`,方法 `rewrite(source: str, plan: OffloadPlan) -> str`。迁移智能体 `MigrationAgent` 只依赖该接口:

- `AstRewriteEngine`(`name="ast"`,**默认**):Python `ast` + 覆盖集自动改写,确定性、离线。
- `LlmEngine`(`name="llm"`):provider 无关的 OpenAI 兼容客户端；CLI 显式选择 LLM 后采用 **strict** 语义，缺配置、调用失败或产物非法都直接失败，不回退 AST。

无论哪种引擎，产物都必须通过同一 callsite 静态门禁、直接 `ap.*` 清单、三层语义验证和 backend capability/provenance 审计。LLM 不能自行声明迁移成功。

引擎由 `orchestrator._make_engine(name)` 选择,CLI 经 `--engine ast|llm` 暴露。

## 后果(Consequences)

**正向**
- 新增引擎 = 新增一个实现类 + 在 `_make_engine` 注册一处,编排器与其它组件零改动(见 QA-1)。
- 默认 AST 保证离线 CI 可复现；显式 LLM 失败不会被另一个引擎的成功掩盖。
- 两种引擎共享同一套契约与验证,便于横向对比。

**负向 / 权衡**
- 抽象带来一层间接;对只需 AST 的最简场景略有冗余。
- LLM 产物仍是完整源码文本，尚未升级成受 schema 约束的 patch/声明协议。

## 备选方案(Alternatives)

- **硬编码 AST-only**:最简,但丧失引擎可替换性与后续 LLM 升级空间——违背课程"可替换构件"目标。
- **配置驱动的映射表**:靠 `{np.op: ap.op}` 字典改写,无法泛化到覆盖集全集,维护成本高。
