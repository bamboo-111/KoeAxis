(() => {
  "use strict";
  const state = {
    workdir: localStorage.getItem("koeaxis_workbench_workdir") || "",
    detail: null,
    review: null,
    recoveryFilter: "all",
    recoveryReasonFilter: "all",
    reviewFilter: "all",
    selectedRecoveryId: "",
    selectedCueId: "",
    job: { status: "idle" },
  };
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  async function request(url, options = {}) {
    const response = await fetch(url, options);
    let payload;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(
        payload?.error?.message ||
          payload?.error ||
          `${response.status} ${response.statusText}`,
      );
    }
    return payload;
  }
  function setNotice(message = "") {
    $("#notice").textContent = message;
  }
  function escapeHtml(value) {
    return String(value ?? "").replace(
      /[&<>"]/g,
      (char) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char],
    );
  }
  function escapeAttr(value) {
    return escapeHtml(value).replace(/'/g, "&#39;");
  }
  function safeClass(value) {
    return String(value)
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-");
  }
  function metric(value, label) {
    return `<div class="metric"><strong>${escapeHtml(String(value ?? "—"))}</strong><span>${escapeHtml(label)}</span></div>`;
  }
  function chip(value) {
    const text = String(value || "unknown");
    return `<span class="status-chip ${safeClass(text)}">${escapeHtml(text)}</span>`;
  }
  function formatTime(ms) {
    if (ms == null) return "—";
    const seconds = Number(ms) / 1000;
    return `${Math.floor(seconds / 60)}:${String((seconds % 60).toFixed(3)).padStart(6, "0")}`;
  }
  function formatBytes(bytes) {
    const value = Number(bytes || 0);
    return value < 1024
      ? `${value} B`
      : value < 1048576
        ? `${(value / 1024).toFixed(1)} KB`
        : `${(value / 1048576).toFixed(1)} MB`;
  }
  function formatDate(seconds) {
    return seconds
      ? new Date(Number(seconds) * 1000).toLocaleString("zh-CN")
      : "—";
  }
  async function loadWorkspaces() {
    const payload = await request("/api/v1/workspaces");
    const items = payload.data || [];
    const select = $("#workspaceSelect");
    select.innerHTML = items.length
      ? items
          .map(
            (item) =>
              `<option value="${escapeAttr(item.workdir)}">${escapeHtml(item.source_name)} · ${escapeHtml(item.name)}</option>`,
          )
          .join("")
      : '<option value="">没有可用工作区</option>';
    if (!items.length) return;
    if (!items.some((item) => item.workdir === state.workdir))
      state.workdir = items[0].workdir;
    select.value = state.workdir;
    localStorage.setItem("koeaxis_workbench_workdir", state.workdir);
  }
  async function refresh() {
    if (!state.workdir) return;
    setNotice("正在刷新工作区状态…");
    try {
      const workdir = encodeURIComponent(state.workdir);
      const [detailPayload, jobPayload] = await Promise.all([
        request(`/api/v1/workspace?workdir=${workdir}`),
        request("/api/v1/job"),
      ]);
      state.detail = detailPayload.data;
      renderAll(jobPayload.data || { status: "idle" });
      setNotice("");
    } catch (error) {
      setNotice(error.message || "加载失败");
    }
  }
  async function refreshJob() {
    try {
      const payload = await request("/api/v1/job");
      renderJob(payload.data || { status: "idle" });
    } catch (error) {
      setNotice(error.message || "任务状态刷新失败");
    }
  }
  function renderAll(job) {
    renderJob(job);
    renderPipeline();
    renderRecovery();
    renderQuality();
    renderExports();
  }
  function renderJob(job) {
    state.job = job;
    const status = String(job.status || "idle");
    const node = $("#jobState");
    node.textContent =
      status === "idle" ? "未运行" : `${job.stage || "任务"} · ${status}`;
    node.className = `status-chip ${safeClass(status)}`;
    $("#stopJobButton").disabled = !["running", "stopping"].includes(status);
  }
  function renderPipeline() {
    const detail = state.detail;
    const stages = detail?.stages?.stages || [];
    const align = detail?.align || {};
    $("#projectSummary").textContent =
      `${detail?.source_name || detail?.name || ""} · ${detail?.workdir || ""}`;
    $("#pipelineMetrics").innerHTML = [
      metric(stages.filter((item) => item.complete).length, "已完成阶段"),
      metric(align.raw_counts?.failed ?? 0, "原始失败"),
      metric(align.dialogue_counts?.failed ?? 0, "对白失败"),
      metric(detail?.quality?.status || "UNKNOWN", "质量门"),
    ].join("");
    const jobBusy = ["running", "stopping"].includes(String(state.job?.status || "idle"));
    $("#stageRows").innerHTML = stages.length
      ? stages
          .map((stage) => {
            const artifact = (stage.artifacts || []).find(
              (item) => item.exists,
            );
            const disabled = jobBusy || !stage.runnable;
            const reason = stage.start_block_reason === "missing_inputs"
              ? `缺少输入：${(stage.missing_inputs || []).join(", ")}`
              : stage.start_block_reason === "managed_by_pipeline"
                ? "该阶段由上游流程内部管理"
                : jobBusy
                  ? "已有任务正在运行"
                  : "继续该阶段";
            const targetView = stageTargetView(stage.name);
            return `<tr><td><strong>${escapeHtml(stage.name)}</strong>${stage.outdated ? '<div class="muted">下游需重算</div>' : ""}</td><td>${chip(stage.status)}</td><td>${stage.input_count ?? "—"}</td><td>${stage.output_count ?? "—"}</td><td>${stage.duration_seconds == null ? "—" : `${stage.duration_seconds.toFixed(1)}s`}</td><td><span class="path-text">${escapeHtml(stage.log?.exists ? stage.log.path : artifact?.path || "无")}</span></td><td><div class="action-row"><button class="button" data-stage-start="${escapeAttr(stage.name)}" title="${escapeAttr(reason)}" ${disabled ? "disabled" : ""}>继续</button>${targetView ? `<button class="button" data-stage-view="${targetView}">查看</button>` : ""}</div></td></tr>`;
          })
          .join("")
      : '<tr><td colspan="7" class="empty-state">没有阶段状态。</td></tr>';
    $$('[data-stage-start]').forEach((button) =>
      button.addEventListener("click", () => startWorkspaceStage(button.dataset.stageStart)),
    );
    $$('[data-stage-view]').forEach((button) =>
      button.addEventListener("click", () => switchView(button.dataset.stageView)),
    );
  }
  function stageTargetView(stage) {
    if (stage === "align") return "recovery";
    if (stage === "quality-gate") return "quality";
    if (stage === "export") return "exports";
    return "";
  }
  function savedStageSettings() {
    try {
      const settings = JSON.parse(localStorage.getItem("koeaxis_webui_settings_v1") || "{}");
      if (!settings || typeof settings !== "object" || Array.isArray(settings)) return {};
      const sensitive = (name) => /(?:api.?key|secret|authorization|access.?token|auth.?token)/i.test(name);
      return Object.fromEntries(Object.entries(settings).filter(([name]) => !sensitive(name)));
    } catch (_) {
      return {};
    }
  }
  async function startWorkspaceStage(stage) {
    if (!stage || !state.workdir) return;
    if (!confirm(`继续运行 ${stage}？已有产物会按 CLI 的 resume 规则处理。`)) return;
    setNotice(`正在启动 ${stage}…`);
    try {
      const payload = await request("/api/v1/workspace/stage/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workdir: state.workdir,
          stage,
          settings: savedStageSettings(),
        }),
      });
      renderJob(payload.data || { status: "running", stage });
      renderPipeline();
      setNotice(`${stage} 已启动。`);
      await refresh();
    } catch (error) {
      setNotice(error.message || `${stage} 启动失败`);
      await refreshJob();
    }
  }
  function filteredRecoveryItems() {
    const items = state.detail?.recovery?.items || [];
    const routed = state.recoveryFilter === "short"
      ? items.filter((item) => item.priority === "short_response")
      : items;
    return state.recoveryReasonFilter === "all"
      ? routed
      : routed.filter((item) => (item.reason_codes || []).includes(state.recoveryReasonFilter));
  }
  function renderRecovery() {
    const recovery = state.detail?.recovery || { items: [] };
    $("#recoveryBadge").textContent = recovery.total || 0;
    $("#recoveryMetrics").innerHTML = [
      metric(state.detail?.align?.dialogue_counts?.completed_exact ?? 0, "精确完成"),
      metric(state.detail?.align?.dialogue_counts?.completed_coarse ?? 0, "粗略完成"),
      metric(state.detail?.align?.dialogue_counts?.failed ?? recovery.total ?? 0, "对白失败"),
      metric(recovery.short_response_count || 0, "短应答"),
      metric(
        state.detail?.align?.excluded_music_region_count || 0,
        "音乐区排除",
      ),
    ].join("");
    const items = filteredRecoveryItems();
    if (!items.some((item) => item.segment_id === state.selectedRecoveryId))
      state.selectedRecoveryId = items[0]?.segment_id || "";
    $("#recoveryList").innerHTML = items.length
      ? items
          .map(
            (item) =>
              `<button class="recovery-item ${item.segment_id === state.selectedRecoveryId ? "active" : ""}" data-recovery-id="${escapeAttr(item.segment_id)}" role="option" aria-selected="${item.segment_id === state.selectedRecoveryId}"><strong>${escapeHtml(item.text || "（空文本）")}</strong><span>${chip(item.priority)}</span><small>${escapeHtml(item.segment_id)} · ${formatTime(item.start_ms)}–${formatTime(item.end_ms)}</small><small>${escapeHtml(item.reason_codes.join(", "))}</small></button>`,
          )
          .join("")
      : '<div class="empty-state">当前筛选没有失败对白。</div>';
    $$("[data-recovery-id]").forEach((button) =>
      button.addEventListener("click", () => {
        state.selectedRecoveryId = button.dataset.recoveryId;
        renderRecovery();
      }),
    );
    renderRecoveryDetail(
      items.find((item) => item.segment_id === state.selectedRecoveryId),
    );
  }
  function mediaUrl(path) {
    return path
      ? `/api/v1/workspace/media?workdir=${encodeURIComponent(state.workdir)}&path=${encodeURIComponent(path)}`
      : "";
  }
  function renderRecoveryDetail(item) {
    const node = $("#recoveryDetail");
    if (!item) {
      node.className = "detail-pane empty-state";
      node.textContent = "选择一个失败片段查看音频、上下文和恢复动作。";
      return;
    }
    node.className = "detail-pane";
    const previous = item.context?.previous;
    const next = item.context?.next;
    const vadRegions = item.vad_proposal?.regions || [];
    const vadSummary = item.vad_proposal
      ? item.vad_proposal.unique_mapping
        ? `VAD 唯一区域：${formatTime(item.vad_proposal.start_ms)}–${formatTime(item.vad_proposal.end_ms)} · ${item.vad_proposal.elapsed_ms ?? "—"}ms`
        : `VAD 返回 ${vadRegions.length} 个区域，必须人工选择一个区域。`
      : "先运行局部 VAD，再决定是否接受 completed_coarse。";
    const regionSelect = vadRegions.length > 1
      ? `<label>VAD 区域<select id="vadRegionIndex">${vadRegions.map((region) => `<option value="${region.index}">${region.index + 1}: ${formatTime(region.start_ms)}–${formatTime(region.end_ms)}</option>`).join("")}</select></label>`
      : "";
    const execution = item.execution
      ? `${item.execution.status || "unknown"} · ${item.execution.strategy || "—"} · ${item.execution.elapsed_ms ?? "—"}ms · ${item.execution.error || "无错误"}`
      : "尚未执行恢复 backend";
    node.innerHTML = `<h2>${escapeHtml(item.text || "（空文本）")}</h2><audio controls preload="metadata" src="${escapeAttr(mediaUrl(item.audio_path))}" aria-label="${escapeAttr(item.segment_id)} 音频"></audio><dl class="detail-grid"><dt>片段</dt><dd>${escapeHtml(item.segment_id)}</dd><dt>状态</dt><dd>${chip(item.status)} ${chip(item.priority)}</dd><dt>时间</dt><dd>${formatTime(item.start_ms)}–${formatTime(item.end_ms)}</dd><dt>失败原因</dt><dd>${escapeHtml(item.error || item.reason_codes.join(", "))}</dd><dt>证据</dt><dd>tokens=${item.token_count} · coverage=${item.coverage ?? "—"}</dd><dt>执行结果</dt><dd>${escapeHtml(execution)}</dd><dt>参考字幕</dt><dd>${item.reference_sources.length ? item.reference_sources.map((source) => escapeHtml(source.name)).join(", ") : "无只读参考源"}</dd></dl><div class="context-strip"><div class="context-item"><small>上一片段</small><div>${escapeHtml(previous?.text || "—")}</div></div><div class="context-item"><small>下一片段</small><div>${escapeHtml(next?.text || "—")}</div></div></div><div class="action-form"><label>核验 transcript<textarea id="verifiedText">${escapeHtml(item.verified_text || item.original_transcript || item.text || "")}</textarea></label><label>语言路由<select id="languageRoute"><option value="Japanese">Japanese</option><option value="English">English</option><option value="Chinese">Chinese</option><option value="auto">auto</option></select></label><label>恢复策略<select id="recoveryStrategy"><option value="auto">auto</option><option value="qwen">qwen</option><option value="mfa-local">mfa-local（日语）</option></select></label><label><input id="useVerifiedText" type="checkbox" ${item.transcript_verified ? "" : "disabled"}> retry 使用已人工核验文本（默认仍用原 transcript）</label>${regionSelect}<div class="action-row"><button class="button" data-action="verify_transcript">保存核验</button><button class="button" data-action="route_language">设置语言</button><button class="button" data-action="localize_vad">局部 VAD</button><button class="button" data-action="retry_align">执行真实重试</button><button class="button" data-action="accept_completed_coarse" ${item.vad_proposal && item.transcript_verified ? "" : "disabled"}>接受粗略完成</button></div><div class="muted">${escapeHtml(vadSummary)}</div></div>`;
    $("#languageRoute").value = item.language_route || item.language || "auto";
    $$("[data-action]").forEach((button) =>
      button.addEventListener("click", () =>
        runRecoveryAction(item, button.dataset.action),
      ),
    );
  }
  async function runRecoveryAction(item, action) {
    const payload = {};
    if (action === "verify_transcript")
      payload.verified_text = $("#verifiedText").value;
    if (action === "route_language")
      payload.language = $("#languageRoute").value;
    if (action === "localize_vad") payload.backend = "pyannote_onnx_v3";
    if (action === "retry_align") {
      payload.strategy = $("#recoveryStrategy").value;
      payload.use_verified_text = $("#useVerifiedText").checked;
    }
    if (action === "accept_completed_coarse" && $("#vadRegionIndex"))
      payload.region_index = Number($("#vadRegionIndex").value);
    if (
      action === "accept_completed_coarse" &&
      !confirm(
        "确认把该片段接受为 completed_coarse？此操作会备份并更新 aligned manifest。",
      )
    )
      return;
    setNotice(`正在执行 ${action}…`);
    try {
      await request("/api/v1/workspace/recovery/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workdir: state.workdir,
          segment_id: item.segment_id,
          action,
          payload,
        }),
      });
      await refresh();
      setNotice(`${action} 已完成。`);
    } catch (error) {
      setNotice(error.message || `${action} 失败`);
    }
  }
  function renderQuality() {
    const quality = state.detail?.quality || {};
    const status = String(quality.status || "UNKNOWN");
    const node = $("#qualityState");
    node.textContent = status;
    node.className = `status-chip ${safeClass(status)}`;
    const summary = quality.summary || {};
    $("#qualitySummary").innerHTML = [
      metric(summary.pass_count ?? "—", "PASS"),
      metric(summary.warn_count ?? "—", "WARN"),
      metric(summary.fail_count ?? "—", "FAIL"),
    ].join("");
    const checks = quality.checks || [];
    $("#qualityChecks").innerHTML = checks.length
      ? checks
          .map(
            (check) => `<div class="evidence-row">
              <strong>${escapeHtml(check.name || check.id || "check")}</strong>
              ${chip(check.status)}
              <span>${escapeHtml(check.message || check.reason || "")}${qualityAction(check)}</span>
            </div>`,
          )
          .join("")
      : `<div class="empty-state">没有分项检查。证据：${escapeHtml(quality.evidence_path || "无")}</div>`;
    $$('[data-quality-view="recovery"]').forEach((button) =>
      button.addEventListener("click", () => switchView("recovery")),
    );
    $$('[data-quality-review]').forEach((button) =>
      button.addEventListener("click", () => openQualityReviewTarget(button.dataset.qualityReview)),
    );
  }
  function qualityAction(check) {
    const status = String(check.status || "").toUpperCase();
    if (!new Set(["FAIL", "WARN"]).has(status)) return "";
    const target = check.target || {};
    if (target.view === "recovery") {
      return ' <button class="button" data-quality-view="recovery">查看恢复项</button>';
    }
    if (target.view === "review" && target.cue_ids?.length) {
      return ` <button class="button" data-quality-review="${escapeAttr(target.cue_ids[0])}">定位 Cue ${escapeHtml(target.cue_ids[0])}</button>`;
    }
    if (target.view === "evidence" && target.path) {
      const href = `/api/v1/workspace/quality-evidence?workdir=${encodeURIComponent(state.workdir)}&path=${encodeURIComponent(target.path)}`;
      return ` <a class="button" href="${escapeAttr(href)}" target="_blank" rel="noopener">打开报告</a>`;
    }
    return "";
  }
  async function openQualityReviewTarget(cueId) {
    state.reviewFilter = "all";
    state.selectedCueId = String(cueId || "");
    $("#reviewFilter").value = "all";
    switchView("review");
    await loadReview();
    renderReview();
    seekSelectedCue();
  }
  function renderExports() {
    const exports = state.detail?.exports || [];
    $("#exportRows").innerHTML = exports.length
      ? exports
          .map((item) => {
            const base = `/api/v1/workspace/export-file?workdir=${encodeURIComponent(state.workdir)}&path=${encodeURIComponent(item.path)}`;
            return `<tr><td><strong>${escapeHtml(item.name)}</strong><div class="path-text">${escapeHtml(item.path)}</div></td><td>${escapeHtml(item.format)}</td><td>${formatBytes(item.size_bytes)}</td><td>${chip(item.delivery_state)}</td><td>${formatDate(item.modified_at)}</td><td><div class="action-row"><a class="button" href="${escapeAttr(base)}" target="_blank" rel="noopener">预览</a><a class="button" href="${escapeAttr(`${base}&download=1`)}">下载</a></div></td></tr>`;
          })
          .join("")
      : '<tr><td colspan="6" class="empty-state">没有导出产物。</td></tr>';
  }
  async function loadReview(force = false) {
    if (!state.workdir || (state.review && !force)) {
      renderReview();
      return;
    }
    setNotice("正在加载 cue 与参考对照…");
    try {
      const payload = await request(
        `/api/v1/workspace/review?workdir=${encodeURIComponent(state.workdir)}`,
      );
      state.review = payload.data;
      renderReview();
      setNotice("");
    } catch (error) {
      setNotice(error.message || "审校数据加载失败");
    }
  }
  function filteredReviewCues() {
    const cues = state.review?.cues || [];
    if (state.reviewFilter === "exact") {
      return cues.filter((cue) => cue.alignment_state === "completed_exact");
    }
    if (state.reviewFilter === "failed") {
      return cues.filter((cue) => cue.alignment_state === "failed");
    }
    if (state.reviewFilter === "coarse") {
      return cues.filter((cue) => cue.alignment_state === "completed_coarse");
    }
    if (state.reviewFilter === "issues") {
      return cues.filter((cue) => (cue.flags || []).length > 0);
    }
    return cues;
  }
  function renderReview() {
    const review = state.review;
    if (!review) return;
    const reviewState = review.review_state || {};
    const allCues = review.cues || [];
    const cues = filteredReviewCues();
    const failed = allCues.filter(
      (cue) => cue.alignment_state === "failed",
    ).length;
    const coarse = allCues.filter(
      (cue) => cue.alignment_state === "completed_coarse",
    ).length;
    const issues = allCues.filter((cue) => (cue.flags || []).length > 0).length;
    $("#reviewMetrics").innerHTML = [
      metric(review.cue_count || 0, "全部 cue"),
      metric(failed, "Align 失败"),
      metric(coarse, "粗略完成"),
      metric(issues, "疑点"),
    ].join("");
    $("#reviewSource").textContent = review.source || "没有 cue 源";
    const draftState = $("#reviewDraftState");
    draftState.textContent = reviewState.dirty
      ? `草稿未应用 · r${reviewState.revision}`
      : reviewState.draft_exists
        ? `草稿已还原 · r${reviewState.revision}`
        : "正式源";
    draftState.className = `status-chip ${reviewState.dirty ? "outdated" : "complete"}`;
    $("#undoReviewButton").disabled = !reviewState.can_undo;
    if (!cues.some((cue) => cue.cue_id === state.selectedCueId)) {
      state.selectedCueId = cues[0]?.cue_id || "";
    }
    $("#reviewCueRows").innerHTML = cues.length
      ? cues
          .map(
            (
              cue,
            ) => `<tr class="review-cue ${cue.cue_id === state.selectedCueId ? "active" : ""}" data-cue-id="${escapeAttr(cue.cue_id)}">
              <td>${formatTime(cue.start_ms)}<br><span class="muted">${formatTime(cue.end_ms)}</span></td>
              <td><div class="cue-text"><strong>${escapeHtml(cue.original || "（空原文）")}</strong><small>${escapeHtml(cue.translation || "（无翻译）")}</small></div></td>
              <td>${chip(cue.alignment_state)}${(cue.flags || []).length ? `<div class="muted">${escapeHtml(cue.flags.join(", "))}</div>` : ""}</td>
            </tr>`,
          )
          .join("")
      : '<tr><td colspan="3" class="empty-state">当前筛选没有 cue。</td></tr>';
    $$("[data-cue-id]").forEach((row) =>
      row.addEventListener("click", () => {
        state.selectedCueId = row.dataset.cueId;
        renderReview();
        seekSelectedCue();
      }),
    );
    const player = $("#reviewPlayer");
    const wantedSource = mediaUrl(review.audio_path);
    if (
      wantedSource &&
      !player.src.includes(encodeURIComponent(review.audio_path))
    ) {
      player.src = wantedSource;
    }
    renderReviewDetail(cues.find((cue) => cue.cue_id === state.selectedCueId));
  }
  function renderReviewDetail(cue) {
    const node = $("#reviewDetail");
    if (!cue) {
      node.className = "review-detail empty-state";
      node.textContent = "选择一条 cue 查看对照。";
      return;
    }
    node.className = "review-detail";
    const references = cue.reference || [];
    node.innerHTML = `<h2>Cue ${escapeHtml(cue.cue_id)}</h2>
      <div class="muted">${formatTime(cue.start_ms)}–${formatTime(cue.end_ms)} · ${escapeHtml(cue.segment_id || "未映射 segment")}</div>
      <div class="flag-list">${chip(cue.alignment_state)}${(cue.flags || []).map((flag) => chip(flag)).join("")}</div>
      <div class="review-edit-form">
        <div class="review-time-fields"><label>开始毫秒<input id="reviewStartMs" type="number" min="0" step="1" value="${escapeAttr(cue.start_ms)}"></label><label>结束毫秒<input id="reviewEndMs" type="number" min="1" step="1" value="${escapeAttr(cue.end_ms)}"></label></div>
        <label>日文原文<textarea id="reviewOriginalText">${escapeHtml(cue.original || "")}</textarea></label>
        <label>翻译<textarea id="reviewTranslationText">${escapeHtml(cue.translation || "")}</textarea></label>
        <div class="action-row"><button id="saveReviewCueButton" class="button">保存到审校草稿</button><span id="reviewEditState" class="muted">正式产物不会被静默覆盖。</span></div>
      </div>
      <h3>只读参考对照</h3>
      ${references.length ? references.map((reference) => `<div class="reference-block"><small>${escapeHtml(reference.source)} · ${escapeHtml(reference.style)} · overlap ${reference.time_overlap_ms}ms</small>${escapeHtml(reference.text)}</div>`).join("") : "<p>当前时间附近没有参考字幕。</p>"}`;
    bindReviewEditor(cue);
  }
  function bindReviewEditor(cue) {
    const fields = [
      $("#reviewStartMs"),
      $("#reviewEndMs"),
      $("#reviewOriginalText"),
      $("#reviewTranslationText"),
    ];
    const status = $("#reviewEditState");
    fields.forEach((field) =>
      field.addEventListener("input", () => {
        status.textContent = "有未保存修改";
        status.className = "status-chip outdated";
      }),
    );
    $("#saveReviewCueButton").addEventListener("click", () => saveReviewCue(cue));
  }
  async function saveReviewCue(cue) {
    const startMs = Number($("#reviewStartMs").value);
    const endMs = Number($("#reviewEndMs").value);
    const original = $("#reviewOriginalText").value.trim();
    const translation = $("#reviewTranslationText").value.trim();
    if (!Number.isInteger(startMs) || !Number.isInteger(endMs) || startMs < 0 || endMs <= startMs) {
      setNotice("时间必须是整数毫秒，并满足 0 ≤ 开始 < 结束。");
      return;
    }
    if (!original) {
      setNotice("日文原文不能为空。");
      return;
    }
    setNotice(`正在保存 Cue ${cue.cue_id}…`);
    try {
      const payload = await request("/api/v1/workspace/review/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workdir: state.workdir,
          cue_id: cue.cue_id,
          original,
          translation,
          start_ms: startMs,
          end_ms: endMs,
          expected_revision: state.review?.review_state?.revision ?? 0,
        }),
      });
      state.review = payload.data.review;
      await refresh();
      renderReview();
      setNotice(payload.data.changed ? "审校草稿已保存；下游质量与导出已标记过期。" : "内容没有变化。");
    } catch (error) {
      setNotice(error.message || "审校草稿保存失败");
    }
  }
  async function undoReviewEdit() {
    if (!state.review?.review_state?.can_undo) return;
    if (!confirm("撤销审校草稿中的上一次编辑？正式 manifest 不会改变。")) return;
    setNotice("正在撤销上一次审校编辑…");
    try {
      const payload = await request("/api/v1/workspace/review/undo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workdir: state.workdir,
          expected_revision: state.review.review_state.revision,
        }),
      });
      state.review = payload.data.review;
      await refresh();
      renderReview();
      setNotice("已撤销上一次审校编辑。");
    } catch (error) {
      setNotice(error.message || "撤销失败");
    }
  }
  function seekSelectedCue() {
    const cue = (state.review?.cues || []).find(
      (item) => item.cue_id === state.selectedCueId,
    );
    const player = $("#reviewPlayer");
    if (!cue || !player.src) return;
    const seek = () => {
      player.currentTime = Math.max(0, cue.start_ms / 1000);
    };
    if (player.readyState >= 1) seek();
    else player.addEventListener("loadedmetadata", seek, { once: true });
  }
  function switchView(name) {
    $$(".nav-item").forEach((button) => {
      const active = button.dataset.view === name;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    $$("[data-view-panel]").forEach((panel) => {
      const active = panel.dataset.viewPanel === name;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
    if (name === "review") loadReview();
  }
  function bindEvents() {
    $$(".nav-item").forEach((button) =>
      button.addEventListener("click", () => switchView(button.dataset.view)),
    );
    $("#workspaceSelect").addEventListener("change", (event) => {
      state.workdir = event.target.value;
      state.selectedRecoveryId = "";
      state.selectedCueId = "";
      state.review = null;
      localStorage.setItem("koeaxis_workbench_workdir", state.workdir);
      refresh();
    });
    $("#recoveryFilter").addEventListener("change", (event) => {
      state.recoveryFilter = event.target.value;
      state.selectedRecoveryId = "";
      renderRecovery();
    });
    $("#recoveryReasonFilter").addEventListener("change", (event) => {
      state.recoveryReasonFilter = event.target.value;
      state.selectedRecoveryId = "";
      renderRecovery();
    });
    $("#reviewFilter").addEventListener("change", (event) => {
      state.reviewFilter = event.target.value;
      state.selectedCueId = "";
      renderReview();
    });
    $("#undoReviewButton").addEventListener("click", undoReviewEdit);
    $("#refreshButton").addEventListener("click", refresh);
    $("#stopJobButton").addEventListener("click", async () => {
      try {
        await request("/api/stop", { method: "POST" });
      } catch (error) {
        setNotice(error.message);
      }
      await refresh();
    });
  }
  async function init() {
    bindEvents();
    try {
      await loadWorkspaces();
      await refresh();
      window.setInterval(refreshJob, 3000);
    } catch (error) {
      setNotice(error.message || "工作台初始化失败");
    }
  }
  document.addEventListener("DOMContentLoaded", init);
})();
