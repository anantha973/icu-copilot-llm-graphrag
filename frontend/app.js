/* ═══════════════════════════════════════════════════════════════════════════
   ICU Clinical Copilot — Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = '/api';
let currentPatientId = null;
let patientsData = [];
let chatHistory = [];
let ws = null;
let isDragging = false;
let startX, startWidth;

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    fetchPatients();
    setInterval(fetchPatients, 15000);
    initWebSocket();
    initSidebarResize();
});

// ── Sidebar resize ───────────────────────────────────────────────────────────
function initSidebarResize() {
    const handle = document.getElementById('resize-handle');
    if (!handle) return;

    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        const w = getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width').trim();
        startWidth = parseInt(w) || 380;
        document.body.style.cursor = 'ew-resize';
        e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        let newW = startWidth + (startX - e.clientX);
        newW = Math.max(280, Math.min(newW, window.innerWidth * 0.5));
        document.documentElement.style.setProperty('--sidebar-width', `${newW}px`);
    });
    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            document.body.style.cursor = '';
        }
    });
}

function toggleSidebar() {
    const sidebar = document.getElementById('ai-sidebar');
    const main = document.getElementById('main-content');
    const btn = document.getElementById('floating-toggle');
    if (sidebar.classList.contains('collapsed')) {
        sidebar.classList.remove('collapsed');
        main.classList.remove('expanded');
        btn.style.display = 'none';
    } else {
        sidebar.classList.add('collapsed');
        main.classList.add('expanded');
        btn.style.display = 'flex';
    }
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function initWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === 'alert') showAlert(`Alert for ${d.patient_id}: ${d.message}`);
    };
    ws.onclose = () => setTimeout(initWebSocket, 5000);
}

function showAlert(msg) {
    const el = document.createElement('div');
    el.className = 'live-alert';
    el.innerHTML = `<span class="material-symbols-outlined">emergency</span> <span>${msg}</span>`;
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 5000);
}

// ── Data fetching ────────────────────────────────────────────────────────────
async function fetchPatients() {
    try {
        const res = await fetch(`${API_BASE}/patients`);
        patientsData = await res.json();
        if (!currentPatientId) renderOverview();
        renderNotifications();  // always update sidebar notifications
    } catch (e) {
        console.error('Failed to fetch patients', e);
    }
}

function getSeverityClass(sev) {
    if (sev === 'RED') return 'badge-red';
    if (sev === 'AMBER') return 'badge-amber';
    return 'badge-green';
}

// ═══════════════════════════════════════════════════════════════════════════
// WAVEFORM RENDERER — generates realistic ECG/SpO2 placeholder traces
// ═══════════════════════════════════════════════════════════════════════════

function drawWaveform(canvas, type, color, severity) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Background grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 9) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Generate waveform points
    const points = [];
    const steps = Math.floor(w / 1.5);

    if (type === 'ecg') {
        // Realistic ECG: flat baseline with periodic QRS complexes
        const cycleLen = Math.floor(steps / 4); // ~4 heartbeats visible
        for (let i = 0; i < steps; i++) {
            const phase = (i % cycleLen) / cycleLen;
            let y;
            if (phase < 0.08) {
                // P wave
                y = 0.5 - 0.08 * Math.sin(phase / 0.08 * Math.PI);
            } else if (phase < 0.15) {
                // flat
                y = 0.5;
            } else if (phase < 0.18) {
                // Q dip
                y = 0.5 + 0.06 * ((phase - 0.15) / 0.03);
            } else if (phase < 0.22) {
                // R peak (tall spike)
                const t = (phase - 0.18) / 0.04;
                y = 0.5 - 0.38 * Math.sin(t * Math.PI);
            } else if (phase < 0.26) {
                // S dip
                const t = (phase - 0.22) / 0.04;
                y = 0.5 + 0.12 * Math.sin(t * Math.PI);
            } else if (phase < 0.45) {
                // flat ST
                y = 0.5;
            } else if (phase < 0.58) {
                // T wave
                y = 0.5 - 0.07 * Math.sin((phase - 0.45) / 0.13 * Math.PI);
            } else {
                // baseline
                y = 0.5;
            }
            // Add tiny noise
            y += (Math.random() - 0.5) * 0.008;
            points.push({ x: (i / steps) * w, y: y * h });
        }
    } else {
        // SpO2 pleth waveform — smooth sawtooth-like pulse
        const cycleLen = Math.floor(steps / 5);
        for (let i = 0; i < steps; i++) {
            const phase = (i % cycleLen) / cycleLen;
            let y;
            if (phase < 0.3) {
                // Systolic upstroke
                const t = phase / 0.3;
                y = 0.7 - 0.4 * Math.pow(t, 0.6);
            } else if (phase < 0.45) {
                // Dicrotic notch
                const t = (phase - 0.3) / 0.15;
                y = 0.3 + 0.12 * Math.sin(t * Math.PI);
            } else {
                // Diastolic runoff
                const t = (phase - 0.45) / 0.55;
                y = 0.3 + 0.4 * Math.pow(t, 0.7);
            }
            y += (Math.random() - 0.5) * 0.005;
            points.push({ x: (i / steps) * w, y: y * h });
        }
    }

    // Draw trace
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    points.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════════════
// ALERT FEED — expandable / minimizable / hidden on detail view
// ═══════════════════════════════════════════════════════════════════════════

let alertFeedMinimized = false;  // user preference: minimized or expanded

function renderNotifications() {
    const feed = document.getElementById('notif-feed');
    const list = document.getElementById('notif-list');
    const count = document.getElementById('notif-count');
    const pill = document.getElementById('alert-badge-pill');
    const pillCount = document.getElementById('alert-badge-count');
    if (!feed || !list || !count) return;

    // Filter to RED and AMBER patients, RED first
    const alertPatients = patientsData
        .filter(p => p.severity === 'RED' || p.severity === 'AMBER')
        .sort((a, b) => (a.severity === 'RED' ? 0 : 1) - (b.severity === 'RED' ? 0 : 1));

    const alertCount = alertPatients.length;
    const hasRed = alertPatients.some(p => p.severity === 'RED');

    // Update counts
    count.textContent = alertCount;
    if (pillCount) pillCount.textContent = alertCount;

    // Update count badge color
    count.style.background = hasRed
        ? 'var(--n-color-status-danger)'
        : 'var(--n-color-status-warning)';

    // Update pill color
    if (pill) {
        if (hasRed) {
            pill.classList.remove('amber');
        } else {
            pill.classList.add('amber');
        }
    }

    // If we're in detail view, hide everything
    if (currentPatientId) {
        feed.classList.add('hidden-detail');
        if (pill) pill.style.display = 'none';
        return;
    }

    // Ward overview: show based on minimized state
    feed.classList.remove('hidden-detail');
    if (alertFeedMinimized) {
        feed.classList.add('minimized');
        if (pill && alertCount > 0) pill.style.display = 'inline-flex';
        else if (pill) pill.style.display = 'none';
    } else {
        feed.classList.remove('minimized');
        if (pill) pill.style.display = 'none';
    }

    // Populate the feed list
    if (alertCount === 0) {
        list.innerHTML = '<div class="notif-empty">All patients stable</div>';
        return;
    }

    list.innerHTML = alertPatients.map(p => {
        const bed = p.demographics.bed || '?';
        const name = p.demographics.name || p.patient_id;
        const sevClass = p.severity === 'RED' ? 'sev-red' : 'sev-amber';
        const dotClass = p.severity === 'RED' ? 'dot-red' : 'dot-amber';
        const v = p.vitals_latest || {};

        // Build concise clinical description
        let desc = '';
        if (p.conditions && p.conditions.length > 0) {
            desc = p.conditions[0].name;
        }
        const flags = [];
        if (v.MAP && v.MAP < 65) flags.push(`MAP ${Math.round(v.MAP)}`);
        if (v.HR && v.HR > 120) flags.push(`HR ${Math.round(v.HR)}`);
        if (v.SpO2 && v.SpO2 < 92) flags.push(`SpO₂ ${Math.round(v.SpO2)}%`);
        if (v.Temp && v.Temp > 38.5) flags.push(`Temp ${v.Temp.toFixed(1)}°C`);
        if (v.RR && v.RR > 25) flags.push(`RR ${Math.round(v.RR)}`);
        if (flags.length > 0) {
            desc += desc ? ' · ' : '';
            desc += flags.join(', ');
        }

        return `<div class="notif-item ${sevClass}" onclick="showPatientDetail('${p.patient_id}')">
            <div class="notif-dot ${dotClass}"></div>
            <div class="notif-body">
                <div class="notif-header-line">
                    <span class="notif-bed">Bed ${bed}</span>
                    <span class="notif-name">${name}</span>
                </div>
                <div class="notif-desc">${desc}</div>
            </div>
        </div>`;
    }).join('');
}

function minimizeAlertFeed() {
    alertFeedMinimized = true;
    const feed = document.getElementById('notif-feed');
    const pill = document.getElementById('alert-badge-pill');
    if (feed) feed.classList.add('minimized');
    if (pill) {
        const alertCount = patientsData.filter(p => p.severity === 'RED' || p.severity === 'AMBER').length;
        pill.style.display = alertCount > 0 ? 'inline-flex' : 'none';
    }
}

function expandAlertFeed() {
    alertFeedMinimized = false;
    const feed = document.getElementById('notif-feed');
    const pill = document.getElementById('alert-badge-pill');
    if (feed) feed.classList.remove('minimized');
    if (pill) pill.style.display = 'none';
}

// ═══════════════════════════════════════════════════════════════════════════
// WARD OVERVIEW RENDERER
// ═══════════════════════════════════════════════════════════════════════════

function renderOverview() {
    const grid = document.getElementById('patients-grid');
    grid.innerHTML = '';

    patientsData.forEach(p => {
        let sevClass = 'sev-green';
        let badgeClass = 'badge-green';
        let insightIcon = 'check_circle';
        let insightText = 'Stable — weaning readiness assessment pending';

        if (p.severity === 'RED') {
            sevClass = 'sev-red';
            badgeClass = 'badge-red';
            insightIcon = 'warning';
            insightText = p.conditions && p.conditions.length > 0
                ? p.conditions[0].name
                : 'High risk — immediate review required';
        } else if (p.severity === 'AMBER') {
            sevClass = 'sev-amber';
            badgeClass = 'badge-amber';
            insightIcon = 'info';
            insightText = p.conditions && p.conditions.length > 0
                ? p.conditions[0].name + ' — requires monitoring'
                : 'Trending — requires monitoring';
        }

        const v = p.vitals_latest || {};
        const hr   = v.HR   || '--';
        const spo2 = v.SpO2 || '--';
        const map  = v.MAP  || '--';
        const rr   = v.RR   || '--';
        const bed  = p.demographics.bed || '?';
        const name = p.demographics.name || p.patient_id;

        const card = document.createElement('div');
        card.className = `monitor-card ${sevClass}`;
        card.onclick = () => showPatientDetail(p.patient_id);

        card.innerHTML = `
            <div class="severity-strip"></div>
            <div class="mc-header">
                <div>
                    <span class="mc-bed">BED ${bed}</span>
                    <span class="mc-name">${name}</span>
                </div>
                <span class="mc-badge ${badgeClass}">${p.severity || 'GREEN'}</span>
            </div>
            <div class="mc-vitals">
                <div class="mc-vital">
                    <div class="mc-vital-label">HR</div>
                    <div class="mc-vital-val v-hr">${typeof hr === 'number' ? Math.round(hr) : hr}<span class="mc-vital-unit">bpm</span></div>
                </div>
                <div class="mc-vital">
                    <div class="mc-vital-label">SpO₂</div>
                    <div class="mc-vital-val v-spo2">${typeof spo2 === 'number' ? Math.round(spo2) : spo2}<span class="mc-vital-unit">%</span></div>
                </div>
                <div class="mc-vital">
                    <div class="mc-vital-label">MAP</div>
                    <div class="mc-vital-val v-map">${typeof map === 'number' ? Math.round(map) : map}<span class="mc-vital-unit">mmHg</span></div>
                </div>
                <div class="mc-vital">
                    <div class="mc-vital-label">RR</div>
                    <div class="mc-vital-val v-rr">${typeof rr === 'number' ? Math.round(rr) : rr}<span class="mc-vital-unit">/m</span></div>
                </div>
            </div>
            <div class="mc-waveform" data-type="ecg" data-sev="${p.severity || 'GREEN'}">
                <canvas></canvas>
            </div>
            <div class="mc-insight">
                <span class="material-symbols-outlined mc-insight-icon">${insightIcon}</span>
                <span>${insightText}</span>
            </div>
        `;

        grid.appendChild(card);
    });

    // Render waveforms after DOM insertion
    requestAnimationFrame(() => {
        document.querySelectorAll('.mc-waveform').forEach(el => {
            const canvas = el.querySelector('canvas');
            if (!canvas) return;
            const sev = el.dataset.sev;
            const idx = Array.from(el.closest('.ward-grid').children).indexOf(el.closest('.monitor-card'));
            const type = idx % 2 === 0 ? 'ecg' : 'spo2';
            const color = type === 'ecg'
                ? getComputedStyle(document.documentElement).getPropertyValue('--vital-hr').trim()
                : getComputedStyle(document.documentElement).getPropertyValue('--vital-spo2').trim();
            drawWaveform(canvas, type, color, sev);
        });
    });

    // Update notification feed
    renderNotifications();
}

// ═══════════════════════════════════════════════════════════════════════════
// VIEW SWITCHING
// ═══════════════════════════════════════════════════════════════════════════

function showOverview() {
    currentPatientId = null;
    document.getElementById('view-detail').style.display = 'none';
    document.getElementById('view-overview').style.display = 'block';
    document.getElementById('chat-history').innerHTML = `<div class="chat-message ai"><span class="chat-label ai-label">Copilot</span> Select a patient from the Ward Overview to begin real-time analysis.</div>`;
    document.getElementById('chat-input').disabled = true;
    document.getElementById('chat-send').disabled = true;
    chatHistory = [];
    renderOverview();
}

async function showPatientDetail(pid) {
    currentPatientId = pid;
    renderNotifications();  // immediately hide alert feed in detail view
    document.getElementById('view-overview').style.display = 'none';
    document.getElementById('view-detail').style.display = 'block';

    // Enable chat
    document.getElementById('chat-history').innerHTML = `<div class="chat-message ai"><span class="chat-label ai-label">Copilot</span> Ready to analyze patient <strong>${pid}</strong>. Knowledge Graph context loaded.</div>`;
    document.getElementById('chat-input').disabled = false;
    document.getElementById('chat-send').disabled = false;
    chatHistory = [];

    try {
        const res = await fetch(`${API_BASE}/patients/${pid}`);
        const data = await res.json();

        // Header
        document.getElementById('detail-name').innerText = `${data.demographics.name || pid} — Bed ${data.demographics.bed || '?'}`;
        const admDate = data.demographics.admission_date ? data.demographics.admission_date.substring(0, 10) : 'Unknown';
        document.getElementById('detail-scenario').innerText = `${data.demographics.age}y ${data.demographics.sex} · Admitted ${admDate}`;

        const badge = document.getElementById('detail-badge');
        badge.className = `status-badge ${getSeverityClass(data.severity)}`;
        badge.innerText = data.severity;

        renderDetailVitals(data);
        renderDetailChart(data);

        // Clinical lists
        const condList = document.getElementById('list-conditions');
        condList.innerHTML = (data.conditions || []).map(c =>
            `<li><strong>${c.name}</strong> <span style="color:var(--n-color-text-weakest)">(${c.status})</span></li>`
        ).join('') || '<li>No active diagnoses</li>';

        const medList = document.getElementById('list-meds');
        medList.innerHTML = (data.medications || []).map(m =>
            `<li><strong>${m.name}</strong> — ${m.dose} ${m.route || ''}</li>`
        ).join('') || '<li>No active medications</li>';

        // Lab results as table rows
        const labBody = document.getElementById('lab-table-body');
        if (labBody) {
            labBody.innerHTML = (data.lab_results || []).map(l => {
                const flag = (l.flag || 'NORMAL').toUpperCase();
                const flagCls = flag === 'HIGH' ? 'flag-high' : flag === 'CRITICAL' ? 'flag-critical' : flag === 'LOW' ? 'flag-low' : 'flag-normal';
                return `<tr>
                    <td>${l.test}</td>
                    <td style="font-weight:600; color:var(--n-color-text)">${l.value}</td>
                    <td>${l.unit}</td>
                    <td><span class="lab-flag ${flagCls}">${flag}</span></td>
                </tr>`;
            }).join('') || '<tr><td colspan="4" style="color:var(--n-color-text-weakest)">No lab results</td></tr>';
        }

        // Knowledge Graph — server returns fully self-contained HTML (cdn_resources=in_line)
        document.getElementById('graph-iframe').srcdoc = '<html><body style="background:#0d1117;margin:0;padding:24px;font-family:Inter,sans-serif;color:#9AA0A6;">Loading Knowledge Graph\u2026</body></html>';
        document.getElementById('graph-meta').textContent = '';
        const gRes = await fetch(`${API_BASE}/graph/${pid}`);
        if (gRes.ok) {
            const gData = await gRes.json();
            document.getElementById('graph-iframe').srcdoc = gData.html;
            document.getElementById('graph-meta').textContent = `${gData.nodes} nodes · ${gData.edges} edges`;
        } else {
            document.getElementById('graph-iframe').srcdoc = '<html><body style="background:#0d1117;margin:0;padding:24px;font-family:Inter,sans-serif;color:#f87171;">Graph not available for this patient.</body></html>';
        }
    } catch (e) {
        console.error('Error fetching patient detail:', e);
    }
}

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(tabId) {
    ['overview', 'knowledge'].forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        if (el) el.style.display = t === tabId ? 'block' : 'none';
    });
    document.querySelectorAll('.detail-tabs .tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
}

// ── Detail vitals ────────────────────────────────────────────────────────────
function renderDetailVitals(data) {
    const grid = document.getElementById('detail-vitals-grid');
    grid.innerHTML = '';
    const v = data.vitals_latest;
    const items = [
        { label: 'MAP',         val: v.MAP,  unit: 'mmHg', isCrit: v.MAP < 65,                cls: 'text-primary' },
        { label: 'Heart Rate',  val: v.HR,   unit: 'bpm',  isCrit: v.HR > 130,                cls: 'text-ecg' },
        { label: 'SpO₂',       val: v.SpO2, unit: '%',    isCrit: v.SpO2 < 90,               cls: 'text-spo2' },
        { label: 'Resp Rate',   val: v.RR,   unit: '/min', isCrit: v.RR > 30,                 cls: 'text-warn' },
        { label: 'Temperature', val: v.Temp, unit: '°C',   isCrit: v.Temp > 39 || v.Temp < 35, cls: 'text-primary' },
    ];

    items.forEach(i => {
        const crit = i.isCrit ? 'critical' : '';
        const color = i.isCrit ? 'text-alert' : i.cls;
        grid.innerHTML += `
            <div class="detail-vital-box ${crit}">
                <span class="vital-label">${i.label}</span>
                <span class="detail-vital-val ${color}">${i.val || '--'}<span class="vital-unit">${i.unit}</span></span>
            </div>`;
    });
}

// ── Detail chart ─────────────────────────────────────────────────────────────
function renderDetailChart(data) {
    if (!data.timeseries || !data.timeseries.timestamps) return;

    const traceMAP = {
        x: data.timeseries.timestamps, y: data.timeseries.MAP,
        name: 'MAP', type: 'scatter', mode: 'lines+markers',
        line: { color: '#38bdf8', width: 2, shape: 'spline' },
        marker: { size: 3 }
    };
    const traceHR = {
        x: data.timeseries.timestamps, y: data.timeseries.HR,
        name: 'HR', type: 'scatter', mode: 'lines+markers',
        line: { color: '#34d399', width: 2, shape: 'spline' },
        marker: { size: 3 }
    };

    Plotly.newPlot('detail-chart', [traceMAP, traceHR], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9AA0A6', family: 'Inter', size: 11 },
        margin: { t: 16, b: 36, l: 40, r: 16 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.04)', zeroline: false },
        yaxis: { gridcolor: 'rgba(255,255,255,0.04)', zeroline: false },
        legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } },
        hovermode: 'x unified'
    }, { responsive: true, displayModeBar: false });
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════════════════════

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg || !currentPatientId) return;

    input.value = '';
    appendMessage('user', msg);

    const responseId = 'ai-' + Date.now();
    appendMessage('ai', '…', responseId);

    try {
        chatHistory.push({ role: 'user', content: msg });

        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: currentPatientId, message: msg, history: chatHistory })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let full = '';
        const el = document.getElementById(responseId);
        el.innerText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            full += decoder.decode(value);

            let html = marked.parse(
                full.replace(/(Source:|Reference:)/g, '<span class="source-label">$1</span>')
                    .replace(/\\geq/g, '≥').replace(/\\leq/g, '≤')
                    .replace(/\\times/g, '×').replace(/\\mu/g, 'μ')
                    .replace(/\\text\{([^}]*)\}/g, '$1').replace(/\$([^$]*)\$/g, '$1')
            );

            if (html.startsWith('<p>')) {
                html = '<p><span class="chat-label ai-label">Copilot</span> ' + html.substring(3);
            } else {
                html = '<span class="chat-label ai-label">Copilot</span> ' + html;
            }
            el.innerHTML = html;

            const hist = document.getElementById('chat-history');
            hist.scrollTop = hist.scrollHeight;
        }

        chatHistory.push({ role: 'assistant', content: full });
    } catch (e) {
        document.getElementById(responseId).innerText = 'Error connecting to AI Copilot.';
    }
}

function handleChatEnter(e) { if (e.key === 'Enter') sendChat(); }

function appendMessage(role, text, id = null) {
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    if (id) div.id = id;

    const label = role === 'ai' ? 'Copilot' : 'You';
    const cls = role === 'ai' ? 'ai-label' : 'user-label';

    let html = text;
    if (typeof marked !== 'undefined') {
        html = marked.parse(text);
        if (html.startsWith('<p>')) {
            html = `<p><span class="chat-label ${cls}">${label}</span> ` + html.substring(3);
        } else {
            html = `<span class="chat-label ${cls}">${label}</span> ` + html;
        }
    } else {
        html = `<span class="chat-label ${cls}">${label}</span> ${text}`;
    }

    div.innerHTML = html;
    const history = document.getElementById('chat-history');
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}
