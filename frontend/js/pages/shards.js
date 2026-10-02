/* 分片管理 Shards */
Components.init('shards');
const C = Components;

let currentJob = '';

async function render() {
  if (!currentJob) return;
  let d;
  try { d = await API.get('/api/jobs/' + currentJob + '/shards'); } catch (e) { return; }

  renderSummary(d);

  const inputs = d.input_shards || [];
  document.getElementById('input-shards').innerHTML = inputs.length
    ? `<div class="small muted mb">共 ${inputs.length} 个分片 shards · 合计 ${C.fmtNum(sum(inputs, 'count'))} 条 / ${C.fmtBytes(sum(inputs, 'size_bytes'))}</div>`
    + inputs.map((s, i) => {
        const max = Math.max(1, ...inputs.map(x => x.size_bytes || 0));
        const pct = Math.round((s.size_bytes || 0) / max * 100);
        const empty = !s.count ? ' <span class="badge muted">空 Empty</span>' : '';
        return `<div style="padding:6px 0;border-bottom:1px solid var(--border)">
           <div class="flex between">
             <span class="mono">${C.esc(s.shard_id)}${empty}</span>
             <span class="muted small tabular">${C.fmtNum(s.count)} 条 · ${C.fmtBytes(s.size_bytes)}</span>
           </div>
           <div class="meter" style="margin-top:3px"><div class="fill ok" style="width:${pct}%"></div></div>
         </div>`;
      }).join('')
    : C.empty();

  document.getElementById('map-tasks').innerHTML = (d.map_tasks || []).length
    ? C.table([
        { key: 'task_id', label: 'Task', render: r => `<span class="mono">${C.esc(r.task_id)}</span>` },
        { key: 'input_shard', label: '分片 Shard', render: r => `<span class="mono">${C.esc(r.input_shard)}</span>` },
        { key: 'status', label: '状态', render: r => C.stateBadge(r.status) },
        { key: 'worker_name', label: 'Worker', render: r => C.esc(r.worker_name || '-') },
      ], d.map_tasks)
    : C.empty();

  document.getElementById('reduce-tasks').innerHTML = (d.reduce_tasks || []).length
    ? C.table([
        { key: 'task_id', label: 'Task', render: r => `<span class="mono">${C.esc(r.task_id)}</span>` },
        { key: 'partition', label: '分区 Partition', render: r => `part-${String(r.partition).padStart(4,'0')}`, num: true },
        { key: 'status', label: '状态', render: r => C.stateBadge(r.status) },
        { key: 'worker_name', label: 'Worker', render: r => C.esc(r.worker_name || '-') },
      ], d.reduce_tasks)
    : C.empty();
}

function sum(rows, key) {
  return rows.reduce((acc, r) => acc + (r[key] || 0), 0);
}

function renderSummary(d) {
  const plan = d.split_plan || {};
  const requested = plan.requested_shards != null ? plan.requested_shards : (d.map_tasks || []).length;
  const actual = plan.actual_shards != null ? plan.actual_shards : (d.input_shards || []).length;
  const clamped = requested !== actual;
  const warnings = plan.warnings || [];

  const chips = [
    `<span class="badge aqua">策略 Strategy: ${C.esc(plan.strategy_label || plan.strategy || '-')}</span>`,
    `<span class="badge ${clamped ? 'warn' : 'good'}">分片数 Shards: ${C.fmtNum(requested)} 请求 → ${C.fmtNum(actual)} 实际</span>`,
    `<span class="badge muted">记录 Records: ${C.fmtNum(plan.total_records)}</span>`,
    `<span class="badge muted">数据量 Size: ${C.fmtBytes(plan.total_bytes)}</span>`,
  ].join(' ');

  const warnHtml = warnings.length
    ? `<ul class="small" style="margin:8px 0 0;padding-left:18px">`
      + warnings.map(w => `<li>⚠️ ${C.esc(w)}</li>`).join('') + `</ul>`
    : '';

  document.getElementById('split-summary').innerHTML = `
    <h2 style="margin-top:0">切分结果 <span class="sub">Split result</span></h2>
    <div class="flex" style="gap:8px;flex-wrap:wrap;align-items:center">${chips}</div>
    ${warnHtml}
    <div class="small muted" style="margin-top:8px">
      策略在提交时确定并固化；下列每个分片即对应一个 Map 任务实际读取处理的数据（条数与字节大小一致）。
      Strategy is locked at submit time; every shard below is exactly the data its map task processes.
    </div>`;
}

C.jobPicker('job-picker', (id) => { currentJob = id; render(); });
C.poll(render, 2000).start();
