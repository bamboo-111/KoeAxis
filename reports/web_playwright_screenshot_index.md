# Web Playwright 截图索引

生成日期：2026-07-16

## 最终验收截图

| 视口 | 页面/场景 | 文件 | 验收重点 |
|---|---|---|---|
| 1440×900 | 流水线阶段操作 | `output/playwright/workbench-stage-actions-1440x900.png` | 11 阶段、继续/查看、禁用态、长证据路径 |
| 1440×900 | 审校编辑器 | `output/playwright/workbench-review-editor-1440x900.png` | 370 cue、时间/原文/翻译草稿、只读参考、固定播放器 |
| 1440×900 | 质量动作 | `output/playwright/workbench-quality-actions-1440x900.png` | 7 个 WARN/FAIL 均有恢复/cue/报告入口 |
| 1440×900 | 质量 cue 定位 | `output/playwright/workbench-quality-cue-target-1440x900.png` | readability 跳 Cue 353 与播放器定位 |
| 1280×720 | 恢复队列 | `output/playwright/final-recovery-1280x720.png` | 22/18/15 指标、列表/详情、音频 ready、无横向溢出 |
| 390×844 | 质量证据 | `output/playwright/final-quality-390x844.png` | 底部导航、长中文、报告按钮、无裁切 |
| 390×844 | 审校总览 | `output/playwright/workbench-review-editor-390x844.png` | 370 cue、长路径、草稿状态、底栏无文字截断 |
| 390×844 | 审校编辑详情 | `output/playwright/workbench-review-editor-detail-390x844.png` | 单列时间输入、文本框、保存按钮、参考字幕、播放器 |
| 390×844 | 恢复队列导航修复 | `output/playwright/workbench-recovery-390x844-fixed.png` | 五个图标按钮完整落在底栏，无裸文本裁切 |
| 1440×900 | 导出面板 | `output/playwright/workbench-exports-1440x900.png` | 5 个产物、quality_gate_failed、预览/下载 |

## 量化浏览器证据

- 1440×900：body 宽度 1425，视口 1440。
- 1280×720：body 宽度 1265，视口 1280。
- 390×844：body 宽度 375，视口 390。
- 移动端 `.nav-label` 与恢复徽标隐藏，但按钮 `aria-label` 保留。
- 370 cue 缓存切换 33.7ms；3.5 秒轮询不替换 cue DOM、不改变行数或 scrollTop=900。
- Console：0 error / 0 warning；失败网络请求：0。
- 音频：206 Partial Content；恢复片段 readyState=4。
- 字幕预览：200、UTF-8，日文/中文正文正确。
- 下载：200、attachment、34034 bytes。

早期无 `-fixed` 后缀的移动截图仅作为问题发现证据；最终验收以本索引列出的修复后截图为准。
