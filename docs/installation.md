# 安装与验证

## 1. 源码工作副本

本发布以源码检出运行，不承诺 `pip install .` 或 wheel 安装。固定链要求 Python 3.12；旧 pyproject 的 `>=3.10` 是旧 CLI 元数据，不适用于固定链资源 profile。

在仓库根目录创建并激活 Python 3.12 环境，安装 `requirements-runtime.txt`，随后：

```sh
python -B tools/build_runtime.py
python -B tools/check_runtime.py
python -B tools/verify_release.py
```

构建脚本仅读取当前环境，不联网、不安装包。它从 `runtime/registry-templates/` 生成被 Git 忽略的 `module-registry/`，包括：

- 精确版本的已安装分发文件摘要与当前解释器 marker；
- 仓库相对的兼容源码根及已固定的源码摘要；
- 当前内容文件、bootstrap 和 catalog 摘要。

这是本机身份绑定，不是 wheel 来源认证或密码学签名。构建前应信任当前源码和安装来源；不要通过重新构建掩盖未知篡改。更换环境、安装额外包或修改受控源码后需重新构建，再运行检查。

`verify_release.py` 用现有 `verify_pkg.py` 顺序运行固定的 20 个测试模块，不使用 discover。继承至子进程的 socket 钩子封锁连接；真实重型依赖在集成测试中使用替身。默认本机验证锁位于 `verify-out/release.lock`，协作环境可用 `KOEAXIS_VERIFY_LOCK` 指向同一锁文件。不要绕过锁并行运行共享输出测试。

失败时保留首次日志，并只重跑失败模块：

```sh
python -B tools/verify_release.py --failed-from verify-out/FIRST --out-dir verify-out/RERUN
```

输出目录必须不存在，防止覆盖首次证据。历史请求档案的两项诊断、历史整期文件的一项预检明确跳过；其他测试不是因为缺档案而跳过。默认历史 LLM 计划的整期 SHA 检查仍失败，测试验证这一事实，不将其伪装为计划通过。

## 2. 旧 CLI / Web 兼容入口

真实旧业务和 Web 界面另需 `requirements.txt` 中的依赖，CI 服务另需 `requirements-ci.txt`。按实际平台安装 PyTorch、FFmpeg 等；本次发布验证没有重新安装或认证这些重型依赖，也没有启动 Web 服务、CI 服务或真实模型。

```sh
python -B main.py --help
python -B webapp.py
```

Windows 的 `start.bat install` 仍安装旧 `requirements.txt`，**不是**固定链初始化器。使用固定链前还必须安装 `requirements-runtime.txt` 并执行构建；若两组依赖发生冲突，应使用独立环境而非忽略版本检查。

若直接执行 `python -m qwen_asr...`，应在 `compat/legacy/` 下执行，或将其加入 `PYTHONPATH`。旧部署示例的相对源码根也对应 `compat/legacy/`。旧工具/测试保持作为兼容源码，不代表它们已加入这次定向验证；实际部署应逐项核对路径、认证、网络与存储配置。

所有新工作目录建议使用显式绝对路径，放在仓库及 `compat/legacy/` 之外。固定链会保护兼容源码目录，不能把模型缓存或产物写入其中。

## 3. 固定链路径检查与真实执行

将 `modules/temporary/real_chain_runner/config.example.json` 复制到仓库外，替换所有占位符：媒体、ASR/对齐模型、模型缓存、说话人权重与 enrollment、节目记忆、MFA 程序/字典/声学模型。节目资源必须按实际节目选择；不要将样例当通用规则。

只检查：

```sh
python -B modules/temporary/real_chain_runner/launch.py --config /absolute/path/to/config.json
```

真实执行需要同时满足：

- `GEMINI_API_KEY`、`MIMO_API_KEY` 由本地环境安全提供；不写入 Git、配置或命令示例。
- `REAL_LLM_STAGES_ONLINE=1`、`REAL_MIMO_ONLINE=1` 明确允许在线阶段。
- 实际端点/模型通过相应环境配置提供。旧默认模型名和服务路由未在此次发布中认证。
- 显式 `--execute`，并选择不存在的输出目录。

```sh
python -B modules/temporary/real_chain_runner/launch.py --config /absolute/path/to/config.json --execute --output runs/experiments/MY-RUN-001
```

这条命令会真正使用模型、工具和在线服务，可能产生费用；它**不属于离线验证步骤**。不要对样例占位路径执行。已有输出目录会被拒绝；此启动器创建新 Host，不提供跨 Host 或崩溃恢复。有限重试能力仅属于同一存活 Host，不可用 `retry=true` 绕过。

## 4. 发布验证限制

本次验证针对 Windows / Python 3.12，使用已安装且逐文件校验的依赖。不是全新联网依赖安装验证，不证明其他平台及真实硬件兼容。模型权重、音频、enrollment、Excel 术语表和私有服务配置需使用者自行准备。真实字幕质量、吞吐、并发模型生命周期与生产部署仍须独立验证。
