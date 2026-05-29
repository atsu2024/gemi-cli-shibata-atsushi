const state = {
  width: 20,
  height: 30,
  radius: 5,
  resolution: 1,
  maxSensors: 200,
  loading: false,
  error: '',
  result: null,
};

function icon(name) {
  const paths = {
    target: '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle>',
    play: '<polygon points="5 3 19 12 5 21 5 3"></polygon>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line>',
    map: '<polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21 3 6"></polygon><line x1="9" y1="3" x2="9" y2="18"></line><line x1="15" y1="6" x2="15" y2="21"></line>',
  };

  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths[name]}</svg>`;
}

function number(value, digits = 2) {
  return Number(value).toLocaleString('ja-JP', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

async function optimizePlacement() {
  state.loading = true;
  state.error = '';
  render();

  try {
    const response = await fetch('/sensor-placement', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        width: state.width,
        height: state.height,
        radius: state.radius,
        resolution: state.resolution,
        maxSensors: state.maxSensors,
      }),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '最適化に失敗しました。');
    state.result = data;
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
    render();
  }
}

function metrics() {
  if (!state.result) {
    return `
      <section class="panel empty">
        <div>${icon('target')}</div>
        <p>条件を入力して配置計算を実行してください。</p>
      </section>
    `;
  }

  const { metrics: m } = state.result;
  return `
    <section class="metric-grid">
      <div class="metric">
        <span>センサ台数</span>
        <strong>${m.sensorCount}</strong>
      </div>
      <div class="metric">
        <span>評価点カバー率</span>
        <strong>${number(m.coverageRatio * 100, 1)}%</strong>
      </div>
      <div class="metric">
        <span>理論下限</span>
        <strong>${m.theoreticalLowerBound}</strong>
      </div>
      <div class="metric">
        <span>未カバー評価点</span>
        <strong>${m.uncoveredSamples}</strong>
      </div>
    </section>
  `;
}

function sensorTable() {
  if (!state.result) return '';

  return `
    <section class="panel">
      <div class="panel-title">
        <h2>配置座標</h2>
        <button class="icon-button" id="download-csv" type="button" title="CSVをダウンロード">
          ${icon('download')}
        </button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>X</th>
              <th>Y</th>
              <th>追加カバー点</th>
            </tr>
          </thead>
          <tbody>
            ${state.result.sensors.map((sensor) => `
              <tr>
                <td>${sensor.id}</td>
                <td>${number(sensor.x)}</td>
                <td>${number(sensor.y)}</td>
                <td>${sensor.coveredSamples}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function render() {
  document.getElementById('root').innerHTML = `
    <div class="app">
      <header class="header">
        <div class="header-mark">${icon('map')}</div>
        <div>
          <h1>センサ配置最適化</h1>
          <p>2D平面を指定半径でカバーする近似最小配置を計算します。</p>
        </div>
      </header>

      <main class="layout">
        <section class="panel controls">
          <h2>条件</h2>
          <form id="placement-form">
            <label>
              <span>エリア幅 m</span>
              <input name="width" type="number" min="0.1" step="0.1" value="${state.width}" />
            </label>
            <label>
              <span>エリア高さ m</span>
              <input name="height" type="number" min="0.1" step="0.1" value="${state.height}" />
            </label>
            <label>
              <span>検知半径 m</span>
              <input name="radius" type="number" min="0.1" step="0.1" value="${state.radius}" />
            </label>
            <label>
              <span>評価分解能 m</span>
              <input name="resolution" type="number" min="0.1" step="0.1" value="${state.resolution}" />
            </label>
            <label>
              <span>最大台数</span>
              <input name="maxSensors" type="number" min="1" step="1" value="${state.maxSensors}" />
            </label>
            <button class="primary-button" type="submit" ${state.loading ? 'disabled' : ''}>
              ${icon('play')}
              ${state.loading ? '計算中' : '配置を計算'}
            </button>
          </form>
          ${state.error ? `<p class="error">${state.error}</p>` : ''}
        </section>

        <section class="panel map-panel">
          <div class="panel-title">
            <h2>配置図</h2>
            <span>${state.width}m x ${state.height}m / 半径 ${state.radius}m</span>
          </div>
          <canvas id="placement-canvas" width="900" height="620"></canvas>
        </section>

        <div class="side">
          ${metrics()}
          ${sensorTable()}
        </div>
      </main>
    </div>
  `;

  bindEvents();
  drawCanvas();
}

function bindEvents() {
  const form = document.getElementById('placement-form');
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const data = new FormData(form);
    state.width = Number(data.get('width'));
    state.height = Number(data.get('height'));
    state.radius = Number(data.get('radius'));
    state.resolution = Number(data.get('resolution'));
    state.maxSensors = Number(data.get('maxSensors'));
    optimizePlacement();
  });

  const downloadButton = document.getElementById('download-csv');
  if (downloadButton) {
    downloadButton.addEventListener('click', downloadCsv);
  }
}

function drawCanvas() {
  const canvas = document.getElementById('placement-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const padding = 48;
  const areaWidth = state.result?.input.width || state.width;
  const areaHeight = state.result?.input.height || state.height;
  const radius = state.result?.input.radius || state.radius;
  const scale = Math.min((width - padding * 2) / areaWidth, (height - padding * 2) / areaHeight);
  const drawWidth = areaWidth * scale;
  const drawHeight = areaHeight * scale;
  const originX = (width - drawWidth) / 2;
  const originY = (height - drawHeight) / 2;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 2;
  ctx.fillRect(originX, originY, drawWidth, drawHeight);
  ctx.strokeRect(originX, originY, drawWidth, drawHeight);

  drawGrid(ctx, originX, originY, drawWidth, drawHeight);

  if (!state.result) {
    ctx.fillStyle = '#64748b';
    ctx.font = '18px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('配置計算後にセンサ範囲を表示します', width / 2, height / 2);
    return;
  }

  state.result.sensors.forEach((sensor) => {
    const x = originX + sensor.x * scale;
    const y = originY + sensor.y * scale;
    ctx.beginPath();
    ctx.arc(x, y, radius * scale, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(14, 165, 233, 0.16)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(2, 132, 199, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  state.result.sensors.forEach((sensor) => {
    const x = originX + sensor.x * scale;
    const y = originY + sensor.y * scale;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#0f766e';
    ctx.fill();
    ctx.fillStyle = '#111827';
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(String(sensor.id), x + 8, y - 8);
  });

  if (state.result.uncoveredPoints.length > 0) {
    ctx.fillStyle = '#dc2626';
    state.result.uncoveredPoints.forEach((point) => {
      ctx.fillRect(originX + point.x * scale - 2, originY + point.y * scale - 2, 4, 4);
    });
  }
}

function drawGrid(ctx, x, y, width, height) {
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  const lines = 10;
  for (let i = 1; i < lines; i += 1) {
    const gx = x + (width / lines) * i;
    const gy = y + (height / lines) * i;
    ctx.beginPath();
    ctx.moveTo(gx, y);
    ctx.lineTo(gx, y + height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + width, gy);
    ctx.stroke();
  }
}

function downloadCsv() {
  if (!state.result) return;
  const rows = [
    ['id', 'x', 'y', 'coveredSamples'],
    ...state.result.sensors.map((sensor) => [sensor.id, sensor.x, sensor.y, sensor.coveredSamples]),
  ];
  const csv = rows.map((row) => row.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'sensor-placement.csv';
  link.click();
  URL.revokeObjectURL(url);
}

render();
optimizePlacement();
