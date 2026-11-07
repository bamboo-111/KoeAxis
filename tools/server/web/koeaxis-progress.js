/* <koeaxis-progress> - framework-free progress widget for a KoeAxis fixed-chain run.
 *
 * Usage:
 *   <koeaxis-progress src="/api/runs/RUN-9/progress" interval="5000"></koeaxis-progress>
 *   or set el.snapshot = {...} yourself (JSON produced by tools/server/progress_snapshot.py).
 *
 * Between polls the elapsed/remaining clocks tick locally from `now_unix_s`, so the display
 * stays smooth. Stage percent and all remaining times are estimates (see snapshot.estimate).
 */
(function () {
  const fmt = (s) => {
    if (s == null || !isFinite(s)) return "--";
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const STYLE = `
    :host { display: block; font: 14px/1.4 system-ui, sans-serif; color: var(--kp-fg, #1f2328); }
    .card { border: 1px solid var(--kp-border, #d0d7de); border-radius: 8px; padding: 14px 16px; background: var(--kp-bg, #fff); }
    .row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
    .title { font-weight: 600; } .muted { color: var(--kp-muted, #656d76); font-size: 12px; }
    .bar { height: 8px; border-radius: 4px; background: var(--kp-track, #eaeef2); overflow: hidden; margin: 6px 0 10px; }
    .fill { height: 100%; background: var(--kp-accent, #0969da); transition: width .6s ease; }
    .fill.overrun { background: var(--kp-warn, #bf8700); }
    .fill.failed { background: var(--kp-bad, #cf222e); } .fill.done { background: var(--kp-ok, #1a7f37); }
    .big { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .steps { display: grid; grid-template-columns: repeat(12, 1fr); gap: 3px; margin-top: 8px; }
    .step { height: 6px; border-radius: 3px; background: var(--kp-track, #eaeef2); }
    .step.done { background: var(--kp-ok, #1a7f37); } .step.running { background: var(--kp-accent, #0969da); }
    .step.failed { background: var(--kp-bad, #cf222e); }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px; }
    .grid div > span { display: block; } .num { font-variant-numeric: tabular-nums; font-weight: 600; }
    .note { margin-top: 8px; }`;

  class KoeAxisProgress extends HTMLElement {
    constructor() {
      super();
      this._snap = null; this._seen = 0; this._timer = null; this._tick = null;
      this.attachShadow({ mode: "open" });
    }
    connectedCallback() {
      this._render();
      this._tick = setInterval(() => this._render(), 1000);
      if (this.getAttribute("src")) { this._poll(); this._timer = setInterval(() => this._poll(), +(this.getAttribute("interval") || 5000)); }
    }
    disconnectedCallback() { clearInterval(this._timer); clearInterval(this._tick); }
    set snapshot(value) { this._snap = value; this._seen = performance.now(); this._render(); }
    get snapshot() { return this._snap; }
    async _poll() {
      try {
        const r = await fetch(this.getAttribute("src"), { cache: "no-store" });
        if (r.ok) this.snapshot = await r.json();
      } catch (_) { this._offline = true; }
    }
    _render() {
      const s = this._snap;
      if (!s) { this.shadowRoot.innerHTML = `<style>${STYLE}</style><div class="card muted">等待进度数据...</div>`; return; }
      const live = s.run_status === "running";
      const drift = live ? (performance.now() - this._seen) / 1000 : 0; // local clock between polls
      const o = s.overall, c = s.current;
      const elapsed = o.elapsed_s + drift;
      const stageElapsed = c ? c.elapsed_s + drift : 0;
      const expired = c && stageElapsed >= c.expected_s;
      const stagePct = c ? (expired ? 99 : Math.min(99, 100 * stageElapsed / Math.max(1, c.expected_s))) : 100;
      const stageLeft = c && c.remaining_s != null && !expired ? Math.max(0, c.remaining_s - drift) : null;
      const totalLeft = o.remaining_s == null || expired ? null : Math.max(0, o.remaining_s - drift);
      const failed = s.run_status === "failed" || s.run_status === "cancelled", done = s.run_status === "completed";
      const cls = failed ? "failed" : done ? "done" : expired ? "overrun" : "";
      const overallPct = done ? 100 : Math.min(99.9, o.percent + (live && totalLeft != null && o.remaining_s ? 100 * drift / (o.elapsed_s + o.remaining_s) : 0));
      const head = done ? "已完成" : s.run_status === "cancelled" ? "已取消" : failed ? "失败" : c ? `${esc(c.label)}  ${c.index}/${c.total}` : "";
      const steps = s.stages.map((x) => `<div class="step ${x.state}" title="${esc(x.label)}: ${x.state}"></div>`).join("");
      this.shadowRoot.innerHTML = `<style>${STYLE}</style><div class="card">
        <div class="row"><span class="title">${head}</span><span class="big">${overallPct.toFixed(1)}%</span></div>
        <div class="bar"><div class="fill ${cls}" style="width:${overallPct}%"></div></div>
        ${c ? `<div class="row"><span class="muted">当前阶段 ${c.index}/${c.total}</span><span class="num">${stagePct.toFixed(0)}%</span></div>
        <div class="bar"><div class="fill ${cls}" style="width:${stagePct}%"></div></div>` : ""}
        <div class="grid">
          <div><span class="muted">已运行</span><span class="num">${fmt(elapsed)}</span></div>
          <div><span class="muted">本阶段剩余</span><span class="num">${c ? (stageLeft == null ? (expired ? "超出预期" : "未知") : "约 " + fmt(stageLeft)) : "--"}</span></div>
          <div><span class="muted">总剩余</span><span class="num">${done ? "0s" : totalLeft == null ? "--" : "约 " + fmt(totalLeft)}</span></div>
        </div>
        <div class="steps">${steps}</div>
        <div class="muted note">阶段百分比与剩余时间仅为估算。依据：${esc(s.estimate.basis)}；时长 ${fmt(s.estimate.duration_s)}；设备 ${esc(s.estimate.device || "未知")}；历史 ${s.estimate.history_runs || 0} 次。云转写缓存及网络耗时未知时不显示倒计时。${this._offline ? " 连接中断，显示的是最后一次数据。" : ""}</div>
      </div>`;
    }
  }
  if (!customElements.get("koeaxis-progress")) customElements.define("koeaxis-progress", KoeAxisProgress);
})();
