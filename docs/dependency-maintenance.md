# Dependency maintenance

This records the local Dependabot consolidation for the 29 original PRs. “Incorporated” means the dependency or workflow change is represented in this branch’s consolidated files; it does not claim that a remote PR has been merged or closed.

| Area | Incorporated PRs | Deferred PRs and reason |
| --- | --- | --- |
| GitHub Actions | #7 `actions/setup-python` 6→7; #8 `actions/setup-node` 6→7 | — |
| Python (`uv`) | #23 `langgraph-api`; #39 `httpx2`; #40 `httpcore2`; #42 grouped non-breaking set; #43 `tornado`; #44 `oauthlib`; #45 `pyjwt`; #46 `urllib3`; #49 `langgraph-sdk`; #50 `pydantic-ai` patch; #55 `mcp` | #17 `pydantic-ai` 2.x is outside the repository’s `<2` extra bound. #19 `google-adk` 2.x is outside its `<2` extra bound. |
| Web (`npm`) | #26 `framer-motion`; #29 `@eslint/js`; #30 TypeScript; #48 grouped non-breaking set; #51 `source-map-js`; #52 `js-yaml`; #53 `browserslist`; #54 `baseline-browser-mapping` | #28 `tailwind-merge` 3.x requires Tailwind 4, while this app remains on Tailwind 3. |
| Demo (`bun`) | #12 `react-day-picker`; #13 `recharts`; #15 `@mdxeditor/editor`; #33 grouped non-breaking set | #14 ESLint 10 is deferred because the Next.js lint plugins currently support ESLint through 9. |

Patch/minor updates and security groups remain enabled for `uv`, npm, and Bun. Dependabot ignores also apply to security updates, so the four explicit major ignores require manual review if a future security fix requires one of those majors; the alerts are not dismissed. See GitHub’s [Dependabot configuration options](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file#ignore). The remaining braces advisory is tracked as upstream-unpatched (`braces <=3.0.3`, GHSA-vfj7-8cjw-p6xm); no patched upstream release is available to consume.

Validation recorded for the consolidated dependency state:

- Web: frozen pnpm install, 3 tests passing, typecheck passing, build passing, and lint passing with 0 errors and 7 warnings.
- Demo: frozen Bun install, TypeScript check, 4 tests passing, and build passing; the resolved `sharp` version is 0.35.4. TypeScript excludes `examples/websocket/**` because those standalone examples import optional `socket.io` packages absent from the app manifest; the app and its tests remain typechecked.
- Python: `uv lock --check` passed. In the locked Python 3.11 environment with `dev`, `adk`, `langgraph`, and `pydantic-ai` extras, the non-integration suite passed 611 tests, skipped 12, and deselected 14; the ADK plus PydanticAI tests passed 31. A separate LangGraph-extra run with `langgraph` 1.2.16 passed 57 tests and skipped 5. An independent Python 3.11 run after optional LangGraph installation passed 612 tests, skipped 11, and deselected 49; the Python 3.14 smoke also passed. CI uses `uv==0.11.7`, `uv sync --locked`, and isolated pinned build/twine tooling.
