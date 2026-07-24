# 测试运行说明

## Python 回归测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

禁用外部 pytest 插件自动加载是为了避免本机 Python 3.14 环境中第三方 `anyio` 插件在测试收集阶段阻塞；Rhein 测试不依赖外部 pytest 插件。

## Cypress 用户行为测试

```bash
npm install
npm run test:e2e
```

该命令会启动本地 Streamlit，再执行 `cypress/e2e/` 的真实浏览器测试。当前 macOS 环境中 Cypress 的 bundled runner 不能处理 `--smoke-test` 参数，因而 Cypress 在启动前失败；这是本机 runner 兼容性阻塞，而不是 app 或测试用例失败。配置和测试用例已提交，需在兼容的本机/CI runner 上执行并纳入阶段验收。
