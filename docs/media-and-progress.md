# 源视频尺寸与预计时间

本次发布仅迁入通用尺寸修复、共用进度估算工具和无框架进度组件。不包含研发目录中的云 ASR 部署整包或新 WebUI 服务；推送 GitHub 不等于更新运行中的服务器。

## ASS

模块化 prepare 通过 PATH 中的 ffprobe 探测一次原媒体，并写入 work/media-source.json（schema 1）：原始媒体绝对路径、秒数、设备和显示尺寸。显示尺寸考虑非方形像素和 90 度倍数旋转，忽略附属封面图片；不支持的旋转会显式失败，避免静默输出错误尺寸。

export 使用该原路径，并在最终质量检查之前同步 work/subtitles.ass 与 work/export/subtitles.ass 的 PlayResX/PlayResY。只改这两个头字段，保持样式、对白、时间轴、BOM 和换行。纯音频以及无此元数据的旧任务保留原行为。既存 ASS 不会被自动批量修改；本修复只影响模块化链，不改变兼容目录中的独立旧 CLI。

源码改变后须重新运行 python -B tools/build_runtime.py 生成本机校验目录。

## ETA

运行 python tools/server/progress_snapshot.py --run-dir <任务输出目录> 可生成 JSON；tools/server/web/koeaxis-progress.js 提供 koeaxis-progress 组件，可通过 snapshot 属性接入该 JSON。本次发布不提供在线 API 路由。

按媒体时长缩放各阶段耗时，读取最近至多 200 个同级目录中的较早已完成任务，以同设备类别的每媒体秒耗时中位数估算；排除当前任务和在当前任务启动之后才完成的样本。CPU/GPU 分组，不区分具体型号。媒体时长来自 sidecar、WAV 头或转写 usage。

云转写尚未完成、媒体时长未知、阶段已超出估算时，不显示倒计时；失败或取消无 ETA，完成时间冻结。缺少阶段样本时 CPU 使用标注低置信度的旧参考；GPU 缺少剩余阶段样本时不提供总倒计时。网络转写耗时不会放大后续计算阶段的速度系数。阶段百分比仍是基于时间的估算，不是真实完成比例；不保证网络服务时延或长短媒体线性。

显式 --baseline 是未按媒体长度缩放的覆盖值，只有确认时长/环境适配时才使用。

## 验证

新增离线 ASS 尺寸与进度回归已纳入 tools/verify_release.py。本次完整既定离线发布套件共 145 项：142 通过、3 项历史档案跳过、0 失败；本机 catalog 重建与校验通过，进度组件 node --check 通过。首次验证的依赖路径与测试夹具问题修正后重跑，不将首次失败计为成功。未以真实在线整链重跑证明 ETA 准确率，未部署生产服务器。
