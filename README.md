# KoeAxis

日语语音转写、对齐、说话人处理、校对与字幕导出工具。本分支将模块化固定链候选与旧 CLI/Web 实现一起发布，保留旧远端提交历史，但不合入本地重构过程中的大体积审计仓库。

## 当前状态

- 模块化入口位于 `modules/temporary/real_chain_runner/launch.py`，默认**只检查**，只有显式 `--execute` 才执行模型、工具和在线阶段。
- `host/` 负责契约、源码捕获和依赖身份校验；`compat/legacy/` 保存固定链仍调用的旧业务实现，不再依赖开发者机器上的旧仓库。
- 根目录 `main.py`、`webapp.py` 保留旧 CLI/Web 入口；它们不是模块化启动器的别名。
- 验证范围是离线定向回归、合成输入及真实 Host 的重型边界替身。**不是生产放行、真实模型兼容认证、完整质量通过或性能改善证明。** 历史质量 FAIL/WARN 不因发布而改变。
- `temporary` 目录名和实验计划名保留其候选身份；并非所有历史迁移切片都在本发布包中。

## 快速开始：模块化候选

使用 Python **3.12**，在仓库根目录执行：

```sh
python -m venv .venv312
# 激活环境；Windows: .venv312\Scripts\activate
# POSIX: source .venv312/bin/activate
python -m pip install -r requirements-runtime.txt
python -B tools/build_runtime.py
python -B tools/verify_release.py
```

`requirements-runtime.txt` 固定离线 Host/替身回归所需版本，不包含真实 ASR、GPU、MFA 的全部依赖。`build_runtime.py` 不下载或安装内容：它从当前安装环境生成本机 `module-registry/`，验证版本、分发文件和仓库源码摘要。安装环境或受控源码变化后应重新构建；运行时仍拒绝摘要不匹配。

验证日志默认写入被忽略的 `verify-out/<唯一编号>/`；三个历史档案专用测试因本包不携带原始档案而明确跳过。无需获取重构仓库的 Git 提交或真实节目产物。

## 使用入口

- 旧 CLI：`python -B main.py --help`
- 旧 Web：`python -B webapp.py`（需额外 Web/模型依赖）
- 固定链检查：复制 `modules/temporary/real_chain_runner/config.example.json` 到本地私有配置，替换全部模型、媒体、工具和资源路径后执行：

```sh
python -B modules/temporary/real_chain_runner/launch.py --config /absolute/path/to/config.json
```

配置示例使用 Windows 绝对路径占位符，其他平台必须改为相应的绝对路径。检查成功只表示路径及资源分发检查通过，不表示模型可运行或服务可用。真正执行还需要明确的在线环境开关、凭据及新的输出目录，详见 `docs/installation.md`。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `host/` | 契约及加载宿主 |
| `modules/temporary/` | 固定链与阶段适配器、评测辅助 |
| `compat/legacy/` | 兼容业务源码、提示词、旧测试/工具及部署示例 |
| `runtime/` | 便携 registry 模板、依赖版本和来源说明 |
| `experiments/specifications/` | 已选择的计划声明 |
| `experiments/references/` | 运行时使用的版本化节目规则 |
| `tests/` | 定向回归和自包含源码基线夹具 |
| `docs/` | 安装与迁移说明 |

不包含模型权重、媒体、声纹嵌入、私有配置、批量 API 请求日志、历史运行产物、内部工单或会话交接。代码运行需要的提示词 `.md` 和第三方许可证不是过程文档，仍保留。

安装和边界见 `docs/installation.md`；与旧布局的差异见 `docs/migration.md`。
