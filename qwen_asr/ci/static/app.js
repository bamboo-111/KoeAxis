/* KoeAxis CI web UI — dependency-free SPA (hash routing). Talks only to the documented /api/v1 endpoints. */
"use strict";

const API = "api/v1";
const STAGE_LABELS = {
  prepare: "准备", transcribe: "识别", correct: "纠错", align: "对齐", "align-recovery": "对齐恢复",
  split: "断句", translate: "翻译", "mimo-proofread": "听音校对", "proofread-realign": "校对重对齐",
  "text-proofread": "文本校对",
  "quality-gate": "质量门", normalize: "规范化", export: "导出",
};
const STATUS_LABELS = {
  queued: "排队", running: "运行中", succeeded: "成功", failed: "失败", cancelled: "已取消",
  completed: "完成", pending: "等待", skipped: "跳过", PASS: "PASS", WARN: "WARN", FAIL: "FAIL",
};
const state = { me: null, meta: null, timers: [] };
const $app = document.getElementById("app");

// ------------------------------------------------------------------ helpers
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const badge = (status) => `<span class="badge ${esc(status)}">${esc(STATUS_LABELS[status] || status || "-")}</span>`;
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString("zh-CN", { hour12: false }) : "-");
const fmtDur = (s) => {
  if (s === null || s === undefined || s === "") return "-";
  s = Number(s);
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m${Math.round(s % 60)}s` : `${Math.floor(m / 60)}h${m % 60}m`;
};
const fmtNum = (n) => (n === null || n === undefined ? "-" : Number(n).toLocaleString("zh-CN"));
const fmtSize = (n) => {
  if (!n && n !== 0) return "-";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};
const runDuration = (run) => {
  if (!run.started_at) return null;
  const end = run.finished_at ? new Date(run.finished_at) : new Date();
  return (end - new Date(run.started_at)) / 1000;
};

function toast(message, error = false) {
  const el = document.createElement("div");
  el.className = "t" + (error ? " error" : "");
  el.textContent = message;
  document.getElementById("toast").appendChild(el);
  setTimeout(() => el.remove(), error ? 7000 : 3500);
}

async function api(path, { method = "GET", body, raw = false } = {}) {
  const headers = { "X-Requested-With": "koeaxis" };
  const init = { method, headers, credentials: "same-origin" };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  const response = await fetch(`${API}${path}`, init);
  if (response.status === 401 && !path.startsWith("/auth/")) { state.me = null; location.hash = "#/login"; throw new Error("请先登录"); }
  if (!response.ok) {
    let detail = response.statusText;
    try { const data = await response.json(); detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail); } catch (_) { /* ignore */ }
    throw new Error(`${response.status} ${detail}`);
  }
  if (raw) return response;
  return response.status === 204 ? null : response.json();
}

function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/blobs`);
    xhr.setRequestHeader("X-Requested-With", "koeaxis");
    xhr.upload.onprogress = (event) => event.lengthComputable && onProgress(event.loaded / event.total);
    xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(`${xhr.status} ${xhr.responseText}`)));
    xhr.onerror = () => reject(new Error("上传失败（网络错误）"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

function every(ms, fn) { const id = setInterval(fn, ms); state.timers.push(id); return id; }
function clearTimers() { state.timers.forEach(clearInterval); state.timers = []; }
const stageBar = (stages) => `<div class="stage-bar" title="阶段进度">${(stages || []).map((s) => `<i class="${esc(s.status)}" title="${esc(STAGE_LABELS[s.name] || s.name)}: ${esc(s.status)}"></i>`).join("")}</div>`;

// ------------------------------------------------------------------ views
async function viewLogin() {
  document.getElementById("nav").hidden = true;
  document.getElementById("user-box").hidden = true;
  $app.innerHTML = `
    <div class="panel login">
      <h1>登录 KoeAxis CI</h1>
      <form id="login-form">
        <label class="field"><span>用户名</span><input name="username" autocomplete="username" required></label>
        <label class="field"><span>密码</span><input name="password" type="password" autocomplete="current-password" required></label>
        <button class="primary" type="submit">登录</button>
      </form>
    </div>`;
  document.getElementById("login-form").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    try {
      await api("/auth/login", { method: "POST", body: { username: form.get("username"), password: form.get("password") } });
      await loadMe();
      location.hash = "#/runs";
    } catch (error) { toast(error.message, true); }
  };
}

async function viewRuns(params) {
  const status = params.get("status") || "";
  const q = params.get("q") || "";
  const preset = params.get("preset") || "";
  const presets = await api("/presets");
  $app.innerHTML = `
    <div class="row"><h1 class="grow">运行</h1><a class="button primary" href="#/new">新建运行</a></div>
    <div class="panel row">
      <select id="f-status">${["", "queued", "running", "succeeded", "failed", "cancelled"].map((s) => `<option value="${s}" ${s === status ? "selected" : ""}>${s ? STATUS_LABELS[s] : "全部状态"}</option>`).join("")}</select>
      <select id="f-preset"><option value="">全部预设</option>${presets.map((p) => `<option value="${esc(p.id)}" ${p.id === preset ? "selected" : ""}>${esc(p.title)}</option>`).join("")}</select>
      <input id="f-q" placeholder="搜索标题 / ID" value="${esc(q)}">
      <label class="row small"><input type="checkbox" id="f-mine" ${params.get("mine") ? "checked" : ""}> 只看我的</label>
      <span class="grow"></span>
      <button id="cmp" disabled>对比选中 (0)</button>
    </div>
    <div id="runs-table"></div>`;
  const apply = () => {
    const next = new URLSearchParams();
    for (const [key, id] of [["status", "f-status"], ["preset", "f-preset"], ["q", "f-q"]]) { const v = document.getElementById(id).value; if (v) next.set(key, v); }
    if (document.getElementById("f-mine").checked) next.set("mine", "1");
    location.hash = `#/runs?${next}`;
  };
  ["f-status", "f-preset", "f-mine"].forEach((id) => (document.getElementById(id).onchange = apply));
  document.getElementById("f-q").onkeydown = (e) => e.key === "Enter" && apply();
  const selected = new Set();
  document.getElementById("cmp").onclick = () => { const [a, b] = [...selected]; location.hash = `#/compare?a=${a}&b=${b}`; };
  const render = async () => {
    const query = new URLSearchParams({ limit: "100" });
    if (status) query.set("status", status);
    if (preset) query.set("preset_id", preset);
    if (q) query.set("q", q);
    if (params.get("mine")) query.set("mine", "true");
    const data = await api(`/runs?${query}`);
    const box = document.getElementById("runs-table");
    if (!box) return;
    if (!data.items.length) { box.innerHTML = `<div class="panel empty">还没有运行。<a href="#/new">新建一个</a></div>`; return; }
    box.innerHTML = `<table><thead><tr><th></th><th>状态</th><th>标题</th><th>预设</th><th>阶段</th><th>质量</th><th>耗时</th><th>创建</th><th>Worker</th></tr></thead><tbody>
      ${data.items.map((run) => `<tr class="clickable" data-id="${esc(run.id)}">
        <td><input type="checkbox" data-pick="${esc(run.id)}" ${selected.has(run.id) ? "checked" : ""}></td>
        <td>${badge(run.status)}${run.queue_position ? `<div class="small muted">第 ${run.queue_position} 位</div>` : ""}</td>
        <td><a href="#/runs/${esc(run.id)}">${esc(run.title)}</a><div class="small muted mono">${esc(run.id)}${run.parent_run_id ? ` ← ${esc(run.parent_run_id)}` : ""}${run.reset_from ? ` · 从 ${esc(STAGE_LABELS[run.reset_from] || run.reset_from)}` : ""}</div>
            <div class="small muted">${esc(run.summary || "")}</div></td>
        <td>${esc(run.preset_id)}</td>
        <td>${stageBar(run.stages)}</td>
        <td>${run.metrics && run.metrics.quality_status ? badge(run.metrics.quality_status) : "-"}</td>
        <td>${fmtDur(runDuration(run))}</td>
        <td class="small">${fmtTime(run.created_at)}<div class="muted">${esc(run.created_by)}</div></td>
        <td class="small">${esc(run.worker_id || "-")}</td></tr>`).join("")}
      </tbody></table><p class="small muted">共 ${data.total} 条</p>`;
    box.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = (e) => { if (e.target.dataset.pick || e.target.tagName === "A") return; location.hash = `#/runs/${tr.dataset.id}`; }));
    box.querySelectorAll("input[data-pick]").forEach((cb) => (cb.onchange = () => {
      cb.checked ? selected.add(cb.dataset.pick) : selected.delete(cb.dataset.pick);
      const btn = document.getElementById("cmp");
      btn.textContent = `对比选中 (${selected.size})`;
      btn.disabled = selected.size !== 2;
    }));
  };
  await render();
  every(5000, () => render().catch(() => {}));
}

async function viewRun(id) {
  let logOffset = 0;
  const run = await api(`/runs/${id}`);
  $app.innerHTML = `<div id="run-head"></div><div id="run-body"></div>
    <h2 class="row"><span class="grow">日志</span><label class="small row"><input type="checkbox" id="follow" checked> 自动滚动</label>
      <a class="small" href="${API}/runs/${esc(id)}/log.txt" download>下载完整日志</a></h2>
    <pre class="log" id="log"></pre>`;
  const renderHead = (r) => {
    const final = ["succeeded", "failed", "cancelled"].includes(r.status);
    document.getElementById("run-head").innerHTML = `
      <div class="row"><h1 class="grow">${esc(r.title)} ${badge(r.status)}</h1>
        ${!final ? `<button class="danger" id="act-cancel">取消</button>` : ""}
        ${final ? `<select id="retry-from"><option value="">从头（复用已完成阶段）</option>${state.meta.retry_stages.map((s) => `<option value="${s}">从「${esc(STAGE_LABELS[s] || s)}」重跑</option>`).join("")}</select>
          <label class="small row"><input type="checkbox" id="retry-fresh"> 全新工作区</label>
          <button class="primary" id="act-retry">重跑</button>
          <button class="danger" id="act-delete">删除</button>` : ""}
      </div>
      <div class="panel"><dl class="kv">
        <dt>ID</dt><dd class="mono">${esc(r.id)}${r.parent_run_id ? ` · 重跑自 <a href="#/runs/${esc(r.parent_run_id)}">${esc(r.parent_run_id)}</a>` : ""}</dd>
        <dt>预设</dt><dd>${esc(r.preset && r.preset.title || r.preset_id)}</dd>
        <dt>素材</dt><dd>${r.media ? `<a href="${API}/runs/${esc(r.id)}/media">${esc(r.media.name)}</a> <span class="muted small">${fmtSize(r.media.size)}</span>` : "-"}</dd>
        <dt>选项</dt><dd class="mono">${esc(JSON.stringify(r.options))}${r.extra_args && r.extra_args.length ? ` · 额外参数 ${esc(r.extra_args.join(" "))}` : ""}</dd>
        <dt>输入文件</dt><dd class="small">${(r.inputs || []).map((i) => `${esc(i.flag)} ${esc(i.name)} <span class="muted">(${esc(i.source)})</span>`).join("<br>") || "-"}</dd>
        <dt>创建</dt><dd>${fmtTime(r.created_at)} · ${esc(r.created_by)}</dd>
        <dt>开始 / 结束</dt><dd>${fmtTime(r.started_at)} → ${fmtTime(r.finished_at)} · ${fmtDur(runDuration(r))}${r.queue_position ? ` · 排队第 ${r.queue_position} 位` : ""}</dd>
        <dt>Worker</dt><dd>${esc(r.worker_id || "-")} · 工作区 <span class="mono">${esc(r.workspace_key)}</span>${r.reset_from ? ` · reset_from=${esc(r.reset_from)}` : ""}</dd>
        <dt>摘要</dt><dd>${esc(r.summary || "-")}${r.error ? `<div class="small" style="color:var(--fail)">${esc(r.error)}</div>` : ""}</dd>
      </dl></div>`;
    const on = (elId, fn) => { const el = document.getElementById(elId); if (el) el.onclick = fn; };
    on("act-cancel", async () => { if (!confirm("确定取消该运行？")) return; await api(`/runs/${id}/cancel`, { method: "POST" }); toast("已请求取消"); refresh(); });
    on("act-retry", async () => {
      const resetFrom = document.getElementById("retry-from").value || null;
      const fresh = document.getElementById("retry-fresh").checked;
      try { const next = await api(`/runs/${id}/retry`, { method: "POST", body: { reset_from: resetFrom, fresh_workspace: fresh } }); location.hash = `#/runs/${next.id}`; } catch (e) { toast(e.message, true); }
    });
    on("act-delete", async () => { if (!confirm("删除运行及其产物？")) return; try { await api(`/runs/${id}`, { method: "DELETE" }); location.hash = "#/runs"; } catch (e) { toast(e.message, true); } });
  };
  const renderBody = (r) => {
    const m = r.metrics || {};
    const cards = [
      ["总耗时", fmtDur(m.wall_time_s ?? runDuration(r))], ["GPU 阶段", fmtDur(m.gpu_phase_s)], ["API 阶段", fmtDur(m.api_phase_s)],
      ["字幕条数", fmtNum(m.subtitle_count)], ["LLM 请求", `${fmtNum(m.llm_requests)}${m.llm_request_failures ? ` / 失败 ${m.llm_request_failures}` : ""}`],
      ["输入 tokens", fmtNum(m.input_tokens)], ["缓存命中", m.cache_hit_rate == null ? "-" : `${(m.cache_hit_rate * 100).toFixed(1)}%`], ["输出 tokens", fmtNum(m.output_tokens)],
    ];
    document.getElementById("run-body").innerHTML = `
      <h2>阶段</h2>
      <div class="timeline">${(r.stages && r.stages.length ? r.stages : state.meta.stages.map((s) => ({ ...s, status: "pending" }))).map((s) => `
        <div class="stage ${esc(s.status)}" title="${esc(s.summary || s.error || "")}">
          <div class="name">${esc(STAGE_LABELS[s.name] || s.name)}</div>
          <div class="meta">${esc(STATUS_LABELS[s.status] || s.status)} · ${fmtDur(s.elapsed_s)}</div>
          ${s.tokens ? `<div class="meta">${fmtNum(s.requests)} 请求 · ${fmtNum(s.tokens.input_tokens)} in</div>` : ""}
          ${s.error ? `<div class="meta" style="color:var(--fail)">${esc(String(s.error).slice(0, 80))}</div>` : ""}
        </div>`).join("")}</div>
      <h2>指标</h2><div class="cards">${cards.map(([k, v]) => `<div class="card"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("")}</div>
      <h2>检查</h2>${(r.checks || []).length ? `<table><tbody>${r.checks.map((c) => `<tr><td style="width:90px">${badge(c.status)}</td><td style="width:220px" class="mono">${esc(c.name)}</td><td>${esc(c.message)}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">运行完成后显示质量门结果。</p>`}
      <h2>产物</h2>${(r.artifacts || []).length ? `<table><tbody>${r.artifacts.map((a) => `<tr><td><a href="${API}/runs/${esc(r.id)}/artifacts/${encodeURIComponent(a.name)}">${esc(a.name)}</a></td><td>${esc(a.kind)}</td><td>${fmtSize(a.size)}</td><td class="small muted">${fmtTime(a.created_at)}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">暂无产物。</p>`}`;
  };
  const pullLog = async () => {
    const data = await api(`/runs/${id}/log?offset=${logOffset}`);
    if (data.text) {
      const el = document.getElementById("log");
      if (!el) return data;
      el.textContent += data.text;
      if (el.textContent.length > 600000) el.textContent = "…(前文已截断，可下载完整日志)\n" + el.textContent.slice(-500000);
      if (document.getElementById("follow").checked) el.scrollTop = el.scrollHeight;
    }
    logOffset = data.next_offset;
    if (data.next_offset < data.size) return pullLog();
    return data;
  };
  let last = run;
  const refresh = async () => {
    last = await api(`/runs/${id}`);
    renderHead(last); renderBody(last);
  };
  renderHead(run); renderBody(run);
  await pullLog();
  every(3000, async () => {
    if (["succeeded", "failed", "cancelled"].includes(last.status) && last._final) return;
    try { await pullLog(); const before = last.status; await refresh(); if (["succeeded", "failed", "cancelled"].includes(last.status)) { last._final = true; await pullLog(); if (before !== last.status) toast(`运行${STATUS_LABELS[last.status]}`); } } catch (_) { /* retry next tick */ }
  });
}

async function viewNew() {
  const presets = await api("/presets");
  const isAdmin = state.me && state.me.role === "admin";
  $app.innerHTML = `
    <h1>新建运行</h1>
    <form id="new-form" class="panel">
      <label class="field"><span>素材（可多选，每个文件一个运行）</span><input type="file" id="media" multiple required accept="audio/*,video/*"></label>
      <div class="form-grid">
        <label class="field"><span>预设</span><select id="preset">${presets.map((p) => `<option value="${esc(p.id)}">${esc(p.title)}</option>`).join("")}</select></label>
        <label class="field"><span>标题（单文件时可改，默认文件名）</span><input id="title" placeholder="默认使用文件名"></label>
        <label class="field"><span>优先级</span><input id="priority" type="number" value="0"></label>
      </div>
      <p class="small muted" id="preset-desc"></p>
      <h2>选项（留空 = 使用预设）</h2>
      <div class="form-grid">
        <label class="field"><span>ASR 模型 (--model)</span><input id="opt-asr_model" placeholder="如 neosophie/Qwen3-ASR-1.7B-JA"></label>
        <label class="field"><span>对齐模型 (--align-model)</span><input id="opt-align_model"></label>
        <label class="field"><span>LLM 模型 (--llm-model)</span><input id="opt-llm_model"></label>
        <label class="field"><span>识别语言 (--language，auto=自动)</span><input id="opt-language"></label>
        <label class="field"><span>目标语言 (--target-language)</span><input id="opt-target_language"></label>
      </div>
      <h2>替换预设文件（可选）</h2>
      <div class="form-grid">${state.meta.input_flags.map((flag) => `<label class="field"><span class="mono">${esc(flag)}</span><input type="file" data-flag="${esc(flag)}"></label>`).join("")}</div>
      ${isAdmin || state.meta.member_extra_args ? `<label class="field"><span>额外 CLI 参数（每行一个，原样追加到 main.py run）</span><textarea id="extra" placeholder="--mimo-nearby-batch-size&#10;3"></textarea></label>` : ""}
      <label class="field"><span>标签（key=value，逗号分隔，便于筛选/对比）</span><input id="labels" placeholder="episode=29, variant=ja-asr"></label>
      <div id="upload-progress"></div>
      <button class="primary" type="submit" id="submit">上传并入队</button>
    </form>`;
  const showDesc = () => {
    const p = presets.find((x) => x.id === document.getElementById("preset").value);
    document.getElementById("preset-desc").textContent = p ? `${p.description} 参数：${p.run_args.join(" ")}；文件：${p.files.map((f) => f.flag + " " + f.name).join("，") || "无"}；默认选项：${JSON.stringify(p.options)}` : "";
  };
  document.getElementById("preset").onchange = showDesc; showDesc();
  document.getElementById("new-form").onsubmit = async (event) => {
    event.preventDefault();
    const files = [...document.getElementById("media").files];
    if (!files.length) return;
    const submit = document.getElementById("submit");
    submit.disabled = true;
    const progress = document.getElementById("upload-progress");
    try {
      const options = {};
      for (const key of ["asr_model", "align_model", "llm_model", "language", "target_language"]) { const v = document.getElementById(`opt-${key}`).value.trim(); if (v) options[key] = v; }
      const inputs = [];
      for (const el of document.querySelectorAll("input[data-flag]")) {
        if (el.files.length) { const blob = await upload(el.files[0], () => {}); inputs.push({ flag: el.dataset.flag, blob_id: blob.id }); }
      }
      const extraEl = document.getElementById("extra");
      const extraArgs = extraEl ? extraEl.value.split("\n").map((s) => s.trim()).filter(Boolean) : [];
      const labels = {};
      document.getElementById("labels").value.split(",").map((s) => s.trim()).filter(Boolean).forEach((pair) => { const [k, ...v] = pair.split("="); labels[k.trim()] = v.join("=").trim(); });
      const created = [];
      for (const file of files) {
        progress.innerHTML = `<p class="small">上传 ${esc(file.name)}（${fmtSize(file.size)}）</p><div class="progress"><i id="bar"></i></div>`;
        const blob = await upload(file, (ratio) => { const bar = document.getElementById("bar"); if (bar) bar.style.width = `${(ratio * 100).toFixed(1)}%`; });
        const run = await api("/runs", { method: "POST", body: {
          preset_id: document.getElementById("preset").value, media_blob_id: blob.id,
          title: files.length === 1 ? document.getElementById("title").value.trim() : "",
          options, inputs, extra_args: extraArgs, labels, priority: Number(document.getElementById("priority").value || 0),
        } });
        created.push(run);
      }
      toast(`已入队 ${created.length} 个运行`);
      location.hash = created.length === 1 ? `#/runs/${created[0].id}` : "#/runs";
    } catch (error) { toast(error.message, true); submit.disabled = false; }
  };
}

async function viewCompare(params) {
  const a = params.get("a") || "", b = params.get("b") || "";
  const runs = (await api("/runs?limit=200&status=succeeded,failed")).items;
  const options = (sel) => runs.map((r) => `<option value="${esc(r.id)}" ${r.id === sel ? "selected" : ""}>${esc(r.title)} · ${esc(r.id)}</option>`).join("");
  $app.innerHTML = `<h1>A/B 对比</h1>
    <div class="panel row"><span>A</span><select id="ca"><option value=""></option>${options(a)}</select><span>B</span><select id="cb"><option value=""></option>${options(b)}</select><button class="primary" id="go">对比</button></div>
    <div id="cmp-body"></div>`;
  document.getElementById("go").onclick = () => { location.hash = `#/compare?a=${document.getElementById("ca").value}&b=${document.getElementById("cb").value}`; };
  if (!a || !b) return;
  const data = await api(`/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  const cell = (s) => (s ? `${badge(s.status)} ${fmtDur(s.elapsed_s)}` : "-");
  const val = (v) => (typeof v === "number" ? fmtNum(Math.round(v * 1000) / 1000) : esc(v ?? "-"));
  const diff = data.srt ? data.srt.diff.map((line) => {
    const cls = line.startsWith("@@") ? "hunk" : line.startsWith("+") && !line.startsWith("+++") ? "add" : line.startsWith("-") && !line.startsWith("---") ? "del" : "";
    return `<span class="${cls}">${esc(line)}</span>`;
  }).join("\n") : "";
  document.getElementById("cmp-body").innerHTML = `
    <table><thead><tr><th></th><th>A：<a href="#/runs/${esc(data.a.id)}">${esc(data.a.title)}</a></th><th>B：<a href="#/runs/${esc(data.b.id)}">${esc(data.b.title)}</a></th></tr></thead><tbody>
      <tr><td>状态</td><td>${badge(data.a.status)}</td><td>${badge(data.b.status)}</td></tr>
      <tr><td>选项</td><td class="mono">${esc(JSON.stringify(data.a.options))}</td><td class="mono">${esc(JSON.stringify(data.b.options))}</td></tr>
      ${data.stages.map((s) => `<tr><td>${esc(STAGE_LABELS[s.name] || s.name)}</td><td>${cell(s.a)}</td><td>${cell(s.b)}</td></tr>`).join("")}
      ${data.metrics.map((m) => `<tr><td class="mono">${esc(m.name)}</td><td>${val(m.a)}</td><td>${val(m.b)}</td></tr>`).join("")}
      ${data.checks.map((c) => `<tr><td class="mono">${esc(c.name)}</td><td>${c.a ? badge(c.a.status) + " " + esc(c.a.message) : "-"}</td><td>${c.b ? badge(c.b.status) + " " + esc(c.b.message) : "-"}</td></tr>`).join("")}
    </tbody></table>
    <h2>字幕差异</h2>
    ${data.srt ? `<p class="small muted">${esc(data.srt.a)}（${data.srt.a_count} 条） vs ${esc(data.srt.b)}（${data.srt.b_count} 条），相似度 ${(data.srt.similarity * 100).toFixed(1)}%${data.srt.truncated ? "（差异过长已截断）" : ""}</p><pre class="diff">${diff || "（无差异）"}</pre>` : `<p class="muted">两边都需要有字幕产物（kind=subtitle）。</p>`}`;
}

async function viewWorkers() {
  const render = async () => {
    const workers = await api("/workers");
    const box = document.getElementById("workers");
    if (!box) return;
    box.innerHTML = workers.length ? `<table><thead><tr><th>ID</th><th>最后心跳</th><th>运行中</th><th>信息</th></tr></thead><tbody>${workers.map((w) => {
      const age = (Date.now() - new Date(w.last_seen_at)) / 1000;
      return `<tr><td><b>${esc(w.name || w.id)}</b><div class="mono small muted">${esc(w.id)}</div></td>
        <td>${badge(age < 120 ? "succeeded" : "failed").replace(/成功|失败/, age < 120 ? "在线" : "离线")} <span class="small muted">${fmtTime(w.last_seen_at)}</span></td>
        <td>${w.running.map((id) => `<a href="#/runs/${esc(id)}" class="mono">${esc(id)}</a>`).join("<br>") || "-"}</td>
        <td class="small mono">${esc(Object.entries(w.info || {}).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  "))}</td></tr>`;
    }).join("")}</tbody></table>` : `<div class="panel empty">还没有 worker 连接。管理员可在「管理 → 令牌」创建 worker 令牌，然后在 GPU 机器上运行 <span class="mono">python -m qwen_asr.ci worker --server … --token …</span></div>`;
  };
  $app.innerHTML = `<h1>Workers</h1><div id="workers"></div>`;
  await render();
  every(10000, () => render().catch(() => {}));
}

async function viewAdmin() {
  if (!state.me || state.me.role !== "admin") { $app.innerHTML = `<p class="muted">仅管理员可见。</p>`; return; }
  const [users, tokens, audit] = await Promise.all([api("/admin/users"), api("/admin/tokens"), api("/admin/audit?limit=100")]);
  $app.innerHTML = `<h1>管理</h1>
    <h2>用户</h2>
    <table><thead><tr><th>用户名</th><th>显示名</th><th>角色</th><th>状态</th><th></th></tr></thead><tbody>${users.map((u) => `<tr>
      <td>${esc(u.username)}</td><td>${esc(u.display_name)}</td><td>${esc(u.role)}</td><td>${u.disabled ? badge("cancelled").replace("已取消", "禁用") : badge("succeeded").replace("成功", "启用")}</td>
      <td><button data-toggle="${u.id}" data-disabled="${u.disabled ? 1 : 0}">${u.disabled ? "启用" : "禁用"}</button> <button data-reset="${u.id}">重置密码</button>
      <button data-role="${u.id}" data-current="${esc(u.role)}">${u.role === "admin" ? "设为成员" : "设为管理员"}</button></td></tr>`).join("")}</tbody></table>
    <form id="new-user" class="panel row" style="margin-top:10px">
      <input name="username" placeholder="用户名" required><input name="display_name" placeholder="显示名"><input name="password" type="password" placeholder="初始密码（≥8 位）" required>
      <select name="role"><option value="member">成员</option><option value="admin">管理员</option></select><button class="primary">添加用户</button></form>
    <h2>令牌</h2>
    <div id="secret"></div>
    <table><thead><tr><th>ID</th><th>名称</th><th>类型</th><th>创建</th><th>最后使用</th><th></th></tr></thead><tbody>${tokens.map((t) => `<tr>
      <td>${t.id}</td><td>${esc(t.name)}</td><td>${esc(t.kind)}</td><td class="small">${fmtTime(t.created_at)}</td><td class="small">${fmtTime(t.last_used_at)}</td>
      <td>${t.revoked ? `<span class="muted">已吊销</span>` : `<button class="danger" data-revoke="${t.id}">吊销</button>`}</td></tr>`).join("")}</tbody></table>
    <form id="new-token" class="panel row" style="margin-top:10px">
      <input name="name" placeholder="名称，如 gpu-01" required><select name="kind"><option value="worker">worker</option><option value="user">user（API 自动化）</option></select>
      <input name="username" placeholder="user 令牌所属用户（默认自己）"><button class="primary">创建令牌</button></form>
    <h2>审计日志</h2>
    <table><tbody>${audit.map((a) => `<tr><td class="small">${fmtTime(a.at)}</td><td>${esc(a.actor)}</td><td class="mono">${esc(a.action)}</td><td class="mono small">${esc(a.target)}</td><td class="mono small">${esc(JSON.stringify(a.detail))}</td></tr>`).join("")}</tbody></table>`;
  const reload = () => viewAdmin().catch((e) => toast(e.message, true));
  $app.querySelectorAll("[data-toggle]").forEach((b) => (b.onclick = async () => { await api(`/admin/users/${b.dataset.toggle}`, { method: "PATCH", body: { disabled: b.dataset.disabled !== "1" } }); reload(); }));
  $app.querySelectorAll("[data-role]").forEach((b) => (b.onclick = async () => { await api(`/admin/users/${b.dataset.role}`, { method: "PATCH", body: { role: b.dataset.current === "admin" ? "member" : "admin" } }); reload(); }));
  $app.querySelectorAll("[data-reset]").forEach((b) => (b.onclick = async () => {
    const password = prompt("新密码（≥8 位）"); if (!password) return;
    try { await api(`/admin/users/${b.dataset.reset}`, { method: "PATCH", body: { password } }); toast("已重置"); } catch (e) { toast(e.message, true); }
  }));
  $app.querySelectorAll("[data-revoke]").forEach((b) => (b.onclick = async () => { if (!confirm("吊销后使用该令牌的 worker/脚本将立即失效")) return; await api(`/admin/tokens/${b.dataset.revoke}`, { method: "DELETE" }); reload(); }));
  document.getElementById("new-user").onsubmit = async (e) => {
    e.preventDefault(); const f = new FormData(e.target);
    try { await api("/admin/users", { method: "POST", body: Object.fromEntries(f) }); toast("已添加"); reload(); } catch (err) { toast(err.message, true); }
  };
  document.getElementById("new-token").onsubmit = async (e) => {
    e.preventDefault(); const f = Object.fromEntries(new FormData(e.target)); if (!f.username) delete f.username;
    try {
      const t = await api("/admin/tokens", { method: "POST", body: f });
      await reload();
      document.getElementById("secret").innerHTML = `<div class="secret">新令牌（只显示这一次，请保存到 worker 的环境变量 KOEAXIS_WORKER_TOKEN）：<div class="mono">${esc(t.token)}</div></div>`;
    } catch (err) { toast(err.message, true); }
  };
}

async function viewAccount() {
  const tokens = await api("/me/tokens");
  $app.innerHTML = `<h1>账号：${esc(state.me.display_name || state.me.username)}</h1>
    <form id="pw" class="panel"><h2>修改密码</h2>
      <label class="field"><span>旧密码</span><input type="password" name="old_password"></label>
      <label class="field"><span>新密码（≥8 位）</span><input type="password" name="new_password" required></label>
      <button class="primary">保存</button></form>
    <div class="panel"><h2>我的 API 令牌</h2><p class="small muted">用于脚本调用 HTTP API：<span class="mono">Authorization: Bearer kxu_…</span>（接口文档见 <a href="api/docs" target="_blank">api/docs</a>）</p>
      <div id="my-secret"></div>
      <table><tbody>${tokens.map((t) => `<tr><td>${esc(t.name)}</td><td class="small">${fmtTime(t.created_at)}</td><td class="small">${fmtTime(t.last_used_at)}</td><td>${t.revoked ? "已吊销" : `<button class="danger" data-revoke="${t.id}">吊销</button>`}</td></tr>`).join("")}</tbody></table>
      <form id="mk" class="row" style="margin-top:10px"><input name="name" placeholder="令牌名称" required><button>创建</button></form></div>`;
  document.getElementById("pw").onsubmit = async (e) => { e.preventDefault(); try { await api("/me/password", { method: "POST", body: Object.fromEntries(new FormData(e.target)) }); toast("密码已修改，请重新登录"); location.hash = "#/login"; } catch (err) { toast(err.message, true); } };
  document.getElementById("mk").onsubmit = async (e) => {
    e.preventDefault();
    try { const t = await api("/me/tokens", { method: "POST", body: Object.fromEntries(new FormData(e.target)) }); await viewAccount(); document.getElementById("my-secret").innerHTML = `<div class="secret">只显示一次：<div class="mono">${esc(t.token)}</div></div>`; } catch (err) { toast(err.message, true); }
  };
  $app.querySelectorAll("[data-revoke]").forEach((b) => (b.onclick = async () => { await api(`/me/tokens/${b.dataset.revoke}`, { method: "DELETE" }); viewAccount(); }));
}

// ------------------------------------------------------------------ router
async function loadMe() {
  const data = await api("/me");
  state.me = data.user;
  state.meta = state.meta || await api("/meta");
  document.getElementById("nav").hidden = false;
  document.getElementById("user-box").hidden = false;
  document.getElementById("user-name").textContent = state.me.display_name || state.me.username;
  document.getElementById("nav-admin").hidden = state.me.role !== "admin";
}

async function route() {
  clearTimers();
  const [path, query] = (location.hash.slice(1) || "/runs").split("?");
  const params = new URLSearchParams(query || "");
  const parts = path.split("/").filter(Boolean);
  if (parts[0] !== "login" && !state.me) {
    try { await loadMe(); } catch (_) { return viewLogin(); }
  }
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === parts[0]));
  try {
    if (parts[0] === "login") return await viewLogin();
    if (parts[0] === "runs" && parts[1]) return await viewRun(parts[1]);
    if (parts[0] === "new") return await viewNew();
    if (parts[0] === "compare") return await viewCompare(params);
    if (parts[0] === "workers") return await viewWorkers();
    if (parts[0] === "admin") return await viewAdmin();
    if (parts[0] === "account") return await viewAccount();
    return await viewRuns(params);
  } catch (error) {
    $app.innerHTML = `<div class="panel"><b>出错了</b><p class="mono">${esc(error.message)}</p></div>`;
  }
}

document.getElementById("logout").onclick = async () => {
  await api("/auth/logout", { method: "POST" }).catch(() => {});
  state.me = null;
  location.hash = "#/login";
};
window.addEventListener("hashchange", route);
route();
