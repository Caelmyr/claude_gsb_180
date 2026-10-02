/* 分片管理 Shards */
Components.init('shards');
const C = Components;

let currentJob = '';

async function render() {
  if (!currentJob) return;
  let d;
  try { d = await API.get('/api/jobs/' + currentJob + '/shards'); } catch (e) { return; }

  const inputs = d.input_shards || [];
  const strategy = d.split_strategy || 'count';
  // Totals are summed from the persisted shard documents themselves, so the
  // page can only ever describe the shards the job actually processed.
  const totalRecords = inputs.reduce((a, s) => a + (s.count || 0), 0);
  const totalBytes = inputs.reduce((a, s) => a + (s.bytes || 0), 0);
  const hasBytes = inputs.some(s => s.bytes != null);

  document.getElementById('shard-summary').innerHTML = `
    <div class="stat-tiles">
      <div class="stat"><div class="label">切分策略 Strategy</div>
        <div class="value" style="font-size:18px">${C.esc(d.split_strategy_label || strategy)}</div></div>
      <div class="stat"><div class="label">分片数 Shards</div><div class="value">${inputs.length}</div></div>
      <div class="stat"><div class="label">总记录 Records</div><div class="value">${C.fmtNum(totalRecords)}</div>
        <div class="delta">输入行数 input rows: ${C.fmtNum(d.input_rows)}</div></div>
      <div class="stat"><div class="label">总大小 Bytes</div><div class="value">${hasBytes ? C.fmtBytes(totalBytes) : '-'}</div></div>
    </div>`;

  // Bar metric follows the strategy: count-balanced jobs compare record
  // counts, size-balanced jobs compare byte volumes.
  const metricOf = s => (strategy === 'size' ? (s.bytes || 0) : (s.count || 0));
  const maxMetric = Math.max(1, ...inputs.map(metricOf));

  document.getElementById('input-shards').innerHTML = inputs.length
    ? `<div class="small muted mb">共 ${inputs.length} 个分片 shards · ${C.fmtNum(totalRecords)} 条 records</div>` +
      inputs.map(s => {
        const pct = Math.round(metricOf(s) / maxMetric * 100);
        const sizeTxt = s.bytes != null ? ` · ${C.fmtBytes(s.bytes)}` : '';
        return `<div style="padding:6px 0;border-bottom:1px solid var(--border)">
           <div class="flex between">
             <span class="mono">${C.esc(s.shard_id)}</span>
             <span class="muted">${C.fmtNum(s.count)} 条${sizeTxt}</span>
           </div>
           <div class="progress" style="margin-top:4px"><div class="fill" style="width:${pct}%"></div></div>
         </div>`;
      }).join('')
    : C.empty();

  document.getElementById('map-tasks').innerHTML = (d.map_tasks || []).length
    ? C.table([
        { key: 'task_id', label: 'Task', render: r => `<span class="mono">${C.esc(r.task_id)}</span>` },
        { key: 'input_shard', label: '分片 Shard', render: r => `<span class="mono">${C.esc(r.input_shard)}</span>` },
        { key: 'status', label: '状态', render: r => C.stateBadge(r.status) },
        { key: 'records_processed', label: '已处理 Records', render: r => C.fmtNum(r.records_processed), num: true },
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

C.jobPicker('job-picker', (id) => { currentJob = id; render(); });
C.poll(render, 2000).start();
