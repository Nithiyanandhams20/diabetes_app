/**
 * DiabetesMeal AI v4.0 — Frontend Application
 * ============================================
 * Handles: navigation, risk assessment, meal plans,
 *          food analyzer, nutrition calculator,
 *          photo scan, NLP chat, food logging.
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const State = {
  selectedHyp:     'no',
  selectedHd:      'no',
  chatHistory:     [],
  allFoods:        [],      // populated from /dataset_stats
  currentSection:  'assess',
};

// ── DOM Ready ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initFoods();
  loadStats();
  loadProfile();
  initDragDrop();
});

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
async function initFoods() {
  try {
    const res   = await fetch('/search_foods');
    const foods = await res.json();
    State.allFoods = foods;
  } catch (e) { console.warn('Could not load food list:', e); }
}

async function loadStats() {
  try {
    const d = await (await fetch('/dataset_stats')).json();
    setText('ds-badge',   `${d.total_records.toLocaleString()} records`);
    setText('s-total',    d.total_records.toLocaleString());
    setText('s-foods',    d.indian_foods_count);
    setText('s-acc',      `🤖 ${d.model_accuracy?.ensemble || '97'}% ML`);
  } catch (e) { console.warn('Stats load failed:', e); }
}

async function loadProfile() {
  try {
    const p = await (await fetch('/get_profile')).json();
    if (p && p.name) {
      const el = document.getElementById('profile-name-display');
      if (el) el.textContent = `👤 ${p.name}`;
    }
  } catch (e) {}
}

// ══════════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════════
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-tab, .tab-pill').forEach(t => t.classList.remove('active'));

  const sec = document.getElementById('sec-' + name);
  if (sec) sec.classList.add('active');
  State.currentSection = name;

  document.querySelectorAll('.nav-tab').forEach(t => {
    if (t.dataset.section === name) t.classList.add('active');
  });
  document.querySelectorAll('.tab-pill').forEach(t => {
    if (t.dataset.section === name) t.classList.add('active');
  });
}

function selectRadio(group, val) {
  if (group === 'hyp') State.selectedHyp = val;
  else                  State.selectedHd  = val;

  ['yes','no'].forEach(v => {
    const el = document.getElementById(`${group}-${v}`);
    if (!el) return;
    if (v === val) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });
}

// ══════════════════════════════════════════════════════════════
// RISK ASSESSMENT
// ══════════════════════════════════════════════════════════════
async function assessRisk() {
  const age     = val('age');
  const glucose = val('glucose');
  const hba1c   = val('hba1c');
  const bmi     = val('bmi');

  if (!age || !glucose || !hba1c || !bmi) {
    showAlert('Please fill in Age, Blood Glucose, HbA1c, and BMI.', 'warning');
    return;
  }

  showLoading('assess-loading');
  hide('risk-result');

  try {
    const d = await post('/predict_diabetes', {
      age, glucose, hba1c, bmi,
      hypertension:  State.selectedHyp === 'yes' ? 1 : 0,
      heart_disease: State.selectedHd  === 'yes' ? 1 : 0,
      smoking:       val('smoking') || 'never',
      gender:        val('gender')  || 'female',
    });

    hideLoading('assess-loading');
    renderRiskResult(d);
  } catch (e) {
    hideLoading('assess-loading');
    showAlert('Error contacting server. Is the Flask app running?', 'danger');
  }
}

function renderRiskResult(d) {
  const el = document.getElementById('risk-result');
  if (!el) return;

  const factors = d.risk_factors.map(f =>
    `<li style="margin:.3rem 0;font-size:.85rem;">⚠️ ${f}</li>`
  ).join('');

  const sim = d.similar_diabetic_pct !== null
    ? `<br><span style="font-size:.8rem;color:var(--muted)">
        ${d.similar_patients} similar patients in dataset —
        ${d.similar_diabetic_pct}% were diabetic
       </span>`
    : '';

  const fiHtml = d.feature_importance
    ? `<div style="margin-top:.75rem;font-size:.78rem;color:var(--muted)">
        <strong>Top risk factors (ML):</strong>
        ${Object.entries(d.feature_importance).map(([k,v]) => `${k}: ${v}%`).join(' | ')}
       </div>`
    : '';

  el.innerHTML = `
    <div class="card-title"><div class="card-icon">📋</div> Risk Assessment Result</div>
    <div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;margin-bottom:1rem;">
      <div style="font-size:3rem;font-weight:900;color:${d.color};line-height:1">${d.risk_score}%</div>
      <div>
        <div style="font-size:1.15rem;font-weight:800;color:${d.color}">${d.risk_level}</div>
        <div style="font-size:.82rem;color:var(--muted)">
          ML Ensemble • ${(d.dataset_size||0).toLocaleString()} patient records •
          ${d.model_accuracy}% accuracy${sim}
        </div>
      </div>
    </div>
    <div class="risk-meter">
      <div class="risk-fill" style="width:0%;background:${d.color}"
           id="risk-fill-bar"></div>
    </div>
    ${factors ? `<ul style="margin-top:.75rem;padding-left:.5rem;">${factors}</ul>` : ''}
    ${fiHtml}
    <div class="info-box ${d.risk_score>=60 ? 'danger' : d.risk_score>=30 ? 'warning' : 'success'}"
         style="margin-top:1rem;">
      ${d.recommendation}
    </div>
    <div class="btn-group">
      <button class="btn btn-primary btn-sm"
        onclick="showSection('meals');setVal('meal-dtype','${d.diabetes_type}');getMealPlan()">
        Get Meal Plan →
      </button>
      <button class="btn btn-secondary btn-sm"
        onclick="quickChat('I have ${d.diabetes_type} diabetes risk, give me South Indian diet advice')">
        Ask AI Assistant →
      </button>
    </div>
  `;
  show('risk-result');
  // Animate bar
  setTimeout(() => {
    const bar = document.getElementById('risk-fill-bar');
    if (bar) bar.style.width = d.risk_score + '%';
  }, 80);
}

function clearAssess() {
  ['age','glucose','hba1c','bmi'].forEach(id => setVal(id, ''));
  selectRadio('hyp', 'no'); selectRadio('hd', 'no');
  hide('risk-result');
}

// ══════════════════════════════════════════════════════════════
// MEAL PLANS
// ══════════════════════════════════════════════════════════════
async function getMealPlan() {
  showLoading('meal-loading');
  setHTML('meal-result', '');

  const dtype   = val('meal-dtype')   || 'type2';
  const glucose = val('meal-glucose') || 120;
  const mtime   = val('meal-time')    || 'all';

  try {
    const d = await post('/get_meal_plan', {
      diabetes_type: dtype,
      glucose_level: parseFloat(glucose),
      meal_time:     mtime,
    });
    hideLoading('meal-loading');
    renderMealPlan(d);
  } catch (e) {
    hideLoading('meal-loading');
    showAlert('Could not load meal plan.', 'danger');
  }
}

function renderMealPlan(d) {
  const icons = { breakfast:'🌅', lunch:'☀️', dinner:'🌙', snacks:'🍎' };
  let html = `<div class="card" style="animation:fadeUp .4s ease">
    <div class="card-title">
      <div class="card-icon">📋</div>
      Meal Plan — ${d.diabetes_type === 'type1' ? 'Type 1' : 'Type 2'} Diabetes
      <span style="margin-left:auto;font-size:.75rem;color:var(--muted)">
        Glucose: ${d.glucose_range || 'normal'} range
      </span>
    </div>`;

  for (const [time, plans] of Object.entries(d.meal_plan)) {
    if (!plans || !plans.length) continue;
    html += `<div class="meal-section">
      <div class="meal-section-title">${icons[time] || '🍴'} ${time.charAt(0).toUpperCase()+time.slice(1)}</div>
      <div class="meal-cards-row">`;

    plans.forEach(m => {
      const gi  = (m.gi_rating || 'low').toLowerCase();
      html += `<div class="meal-card">
        <h4>${m.meal_name || m.name || ''}</h4>
        <p>${m.reason || ''}</p>
        <div class="flex gap-1 flex-wrap" style="margin-top:.5rem;">
          <span class="gi-badge gi-${gi}">${m.gi_icon || ''} ${m.gi_rating || ''} GI</span>
          <span class="gi-badge gi-low">~${Math.round(m.total_calories||200)} kcal</span>
        </div>
        ${m.foods_display ? `<p class="text-xs text-muted mt-1">${m.foods_display}</p>` : ''}
      </div>`;
    });
    html += '</div></div>';
  }

  if (d.foods_to_avoid && d.foods_to_avoid.length) {
    html += `<div style="margin-top:1rem;">
      <div style="font-size:.82rem;font-weight:700;color:var(--red);margin-bottom:.5rem;">🚫 Foods to Avoid</div>
      <div class="avoid-tags">${d.foods_to_avoid.map(f => `<span class="avoid-tag">${f}</span>`).join('')}</div>
    </div>`;
  }
  html += '</div>';
  setHTML('meal-result', html);
}

// ══════════════════════════════════════════════════════════════
// FOOD ANALYZER
// ══════════════════════════════════════════════════════════════
function showFoodSuggestions(inputVal) {
  const box = document.getElementById('food-suggestions');
  if (!box || !inputVal || inputVal.length < 2) { if (box) box.style.display='none'; return; }

  const q = inputVal.toLowerCase();
  const matches = State.allFoods.filter(f =>
    f.name.replace(/_/g,' ').includes(q) ||
    (f.name_local && f.name_local.toLowerCase().includes(q))
  ).slice(0, 9);

  if (!matches.length) { box.style.display='none'; return; }

  box.innerHTML = matches.map(f => {
    const display = f.name.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
    const local   = f.name_local ? `<span style="color:var(--muted)">${f.name_local}</span>` : '';
    const giCls   = `gi-${f.gi}`;
    return `<div class="autocomplete-item" onclick="selectFood('${f.name}','${display}')">
      <span>${display} ${local}</span>
      <span class="gi-tag ${giCls}" style="padding:.15rem .4rem;border-radius:8px;font-size:.68rem;font-weight:700">
        GI ${f.gi_value || '?'}
      </span>
    </div>`;
  }).join('');
  box.style.display = 'block';
}

function selectFood(key, display) {
  setVal('food-name', display);
  const box = document.getElementById('food-suggestions');
  if (box) box.style.display = 'none';
}

document.addEventListener('click', e => {
  if (!e.target.closest('.food-search-wrap')) {
    const box = document.getElementById('food-suggestions');
    if (box) box.style.display = 'none';
  }
});

async function analyzeFood() {
  const name  = val('food-name');
  const port  = val('portion') || 100;
  const dtype = val('analyze-dtype') || 'type2';
  if (!name) { showAlert('Please enter a food name.', 'warning'); return; }

  showLoading('analyze-loading');
  hide('analyze-result');

  try {
    const d = await post('/analyze_food', {
      food_name: name, portion_g: parseFloat(port), diabetes_type: dtype,
    });
    hideLoading('analyze-loading');
    renderFoodResult(d, 'analyze-result');
  } catch (e) {
    hideLoading('analyze-loading');
    showAlert('Error analyzing food.', 'danger');
  }
}

function renderFoodResult(d, targetId) {
  const el = document.getElementById(targetId);
  if (!el) return;

  if (!d.found) {
    el.innerHTML = `<div class="info-box warning">⚠️ ${d.message || 'Food not found. Check spelling or use the autocomplete.'}</div>`;
    show(targetId);
    return;
  }

  const local = d.name_local ? ` <span class="text-muted">(${d.name_local})</span>` : '';
  el.innerHTML = `
    <div class="card-title">
      <div class="card-icon">📊</div>
      ${d.food}${local} — ${d.portion_g}g
    </div>
    <div class="info-box ${d.suitability.includes('✅') ? 'success' : 'warning'}"
         style="border-color:${d.suitability_color};margin-bottom:1rem;">
      <strong>${d.suitability}</strong><br>
      <span style="font-size:.83rem">${d.advice}</span>
    </div>
    <div class="nutrition-grid">
      <div class="nutrition-item"><div class="val">${d.calories}</div><div class="lbl">Calories (kcal)</div></div>
      <div class="nutrition-item"><div class="val">${d.glucose_impact}g</div><div class="lbl">Glucose Impact</div></div>
      <div class="nutrition-item"><div class="val">${d.carbs}g</div><div class="lbl">Carbohydrates</div></div>
      <div class="nutrition-item"><div class="val">${d.protein}g</div><div class="lbl">Protein</div></div>
      <div class="nutrition-item"><div class="val">${d.fat}g</div><div class="lbl">Fat</div></div>
      <div class="nutrition-item"><div class="val">${d.fiber}g</div><div class="lbl">Fiber</div></div>
    </div>
    <div class="mt-2 text-sm text-muted">
      GI: <strong>${(d.gi||'?').toUpperCase()}</strong> (score: ${d.gi_value || '?'}) •
      Category: ${d.category || '?'} •
      Region: ${(d.region||'').replace(/_/g,' ')}
      ${d.notes ? `<br><em>📝 ${d.notes}</em>` : ''}
    </div>
  `;
  show(targetId);
}

// ══════════════════════════════════════════════════════════════
// NUTRITION CALCULATOR
// ══════════════════════════════════════════════════════════════
function setCalcExample(txt) { setVal('calc-input', txt); }

async function calcNutrition() {
  const meal = val('calc-input');
  if (!meal) { showAlert('Please describe your meal first.', 'warning'); return; }

  showLoading('calc-loading');
  hide('calc-result');

  try {
    const d = await post('/nutrition_calc', { meal_text: meal });
    hideLoading('calc-loading');
    renderCalcResult(d);
  } catch (err) {
    hideLoading('calc-loading');
    const msg = err.message || 'No recognisable foods found. Try: "3 idli + sambar + coconut chutney"';
    const res = document.getElementById('calc-result');
    res.innerHTML = `<div class="info-box warning">⚠️ ${msg}</div>`;
    show('calc-result');
  }
}

function renderCalcResult(d) {
  const t   = d.totals;
  const warn = t.glc > 80 ? '🚨 Very high glucose load' :
               t.glc > 45 ? '⚠️ Moderate glucose load' :
               '✅ Manageable for diabetics';

  const rows = d.breakdown.map(r => `
    <tr>
      <td class="fw-bold">${r.food}</td>
      <td class="text-muted">${r.grams}g</td>
      <td>${r.cal}</td>
      <td>${r.glc}</td>
      <td>${r.carb}</td>
      <td>${r.pro}</td>
      <td>${r.fat}</td>
      <td>${r.fib}</td>
      <td><span class="gi-badge gi-${r.gi}">${(r.gi||'').toUpperCase()}</span></td>
    </tr>`).join('');

  const el = document.getElementById('calc-result');
  el.innerHTML = `
    <div class="card-title">
      <div class="card-icon">🧮</div>
      Meal Breakdown — ${d.items_found} food${d.items_found>1?'s':''}
    </div>
    <table class="calc-table">
      <thead><tr>
        <th>Food</th><th>Portion</th><th>Cal</th><th>Glucose</th>
        <th>Carbs</th><th>Protein</th><th>Fat</th><th>Fiber</th><th>GI</th>
      </tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr>
        <td>TOTAL</td><td>—</td>
        <td>${t.cal}</td><td>${t.glc}</td>
        <td>${t.carb}</td><td>${t.pro}</td>
        <td>${t.fat}</td><td>${t.fib}</td><td>—</td>
      </tr></tfoot>
    </table>
    <div class="nutrition-grid" style="margin-top:0">
      <div class="nutrition-item"><div class="val">${t.cal}</div><div class="lbl">Total Calories</div></div>
      <div class="nutrition-item"><div class="val">${t.glc}g</div><div class="lbl">Glucose ${warn}</div></div>
      <div class="nutrition-item"><div class="val">${t.carb}g</div><div class="lbl">Carbohydrates</div></div>
      <div class="nutrition-item"><div class="val">${t.pro}g</div><div class="lbl">Protein</div></div>
      <div class="nutrition-item"><div class="val">${t.fat}g</div><div class="lbl">Fat</div></div>
      <div class="nutrition-item"><div class="val">${t.fib}g</div><div class="lbl">Fiber</div></div>
    </div>
  `;
  show('calc-result');
}

// ══════════════════════════════════════════════════════════════
// PHOTO SCANNER
// ══════════════════════════════════════════════════════════════
function initDragDrop() {
  const zone = document.getElementById('upload-zone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) previewImage({ files:[file] });
  });
}

function previewImage(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    if (img) { img.src = e.target.result; img.style.display = 'block'; }
  };
  reader.readAsDataURL(file);
}

async function analyzePhoto() {
  const file  = document.getElementById('food-image')?.files[0];
  const dtype = val('photo-dtype') || 'type2';
  if (!file) { showAlert('Please upload a food photo first.', 'warning'); return; }

  showLoading('photo-loading');
  hide('photo-result');

  const b64 = await fileToBase64(file);
  try {
    const d = await post('/analyze_image', { image_data: b64, diabetes_type: dtype });
    hideLoading('photo-loading');
    renderPhotoResult(d);
  } catch (e) {
    hideLoading('photo-loading');
    showAlert('Image analysis failed. Try the Food Analyzer tab.', 'danger');
  }
}

function renderPhotoResult(d) {
  if (!d.success) {
    const el = document.getElementById('photo-result');
    el.innerHTML = `<div class="info-box danger">❌ ${d.error || d.note || 'Analysis failed.'}</div>`;
    show('photo-result');
    return;
  }
  const n   = d.nutrition || {};
  const conf= d.confidence;
  const confColor = conf==='high' ? 'var(--green)' : conf==='medium' ? 'var(--amber)' : 'var(--muted)';
  const alts = (d.alternatives||[]).map(a => `${a[0]} (${Math.round(a[1]*100)}%)`).join(', ');

  const el = document.getElementById('photo-result');
  el.innerHTML = `
    <div class="card-title"><div class="card-icon">🔍</div> Photo Analysis Result</div>
    <div class="flex items-center gap-2 flex-wrap" style="margin-bottom:1rem;">
      <strong style="font-size:1.2rem">${d.detected_food}</strong>
      <span style="background:${confColor}22;color:${confColor};padding:.2rem .7rem;
            border-radius:20px;font-size:.75rem;font-weight:700">${conf} confidence</span>
    </div>
    ${alts ? `<p class="text-sm text-muted mb-2">Also possible: ${alts}</p>` : ''}
    ${d.advice ? `<div class="info-box ${d.advice.includes('✅')?'success':'warning'}" style="margin-bottom:1rem">${d.advice}</div>` : ''}
    <div class="nutrition-grid">
      <div class="nutrition-item"><div class="val">${n.calories||'?'}</div><div class="lbl">Calories (kcal)</div></div>
      <div class="nutrition-item"><div class="val">${n.glucose_impact||'?'}g</div><div class="lbl">Glucose Impact</div></div>
      <div class="nutrition-item"><div class="val">${n.carbs||'?'}g</div><div class="lbl">Carbs</div></div>
      <div class="nutrition-item"><div class="val">${n.protein||'?'}g</div><div class="lbl">Protein</div></div>
      <div class="nutrition-item"><div class="val">${n.fat||'?'}g</div><div class="lbl">Fat</div></div>
      <div class="nutrition-item"><div class="val">${n.fiber||'?'}g</div><div class="lbl">Fiber</div></div>
    </div>
    <p class="text-xs text-muted mt-2">📍 ${d.note || ''}</p>
  `;
  show('photo-result');
}

// ══════════════════════════════════════════════════════════════
// NLP CHAT
// ══════════════════════════════════════════════════════════════
function quickChat(txt) {
  setVal('chat-input', txt);
  sendChat();
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg   = (input?.value || '').trim();
  if (!msg) return;
  input.value = '';

  const chat = document.getElementById('chat-messages');
  appendChatMsg(chat, 'user', escapeHTML(msg));

  const typingId = 'typing-' + Date.now();
  appendChatMsg(chat, 'bot',
    '<div class="spinner" style="width:18px;height:18px;border-width:2px;margin:0"></div>',
    typingId);
  scrollChat(chat);

  try {
    const d = await post('/chat', { message: msg, history: State.chatHistory });
    const reply = d.reply || '🤔 No response. Please try again.';

    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    appendChatMsg(chat, 'bot', formatChatReply(reply));
    State.chatHistory.push({ role: 'user',      content: msg   });
    State.chatHistory.push({ role: 'assistant', content: reply });

    // Keep history bounded
    if (State.chatHistory.length > 20) {
      State.chatHistory = State.chatHistory.slice(-16);
    }
  } catch (e) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.innerHTML = '<div class="chat-bubble">⚠️ Server error. Is the Flask app running?</div>';
  }
  scrollChat(chat);
}

function appendChatMsg(chat, role, html, id) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  if (id) div.id = id;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.innerHTML = html;
  div.appendChild(bubble);
  chat.appendChild(div);
}

function formatChatReply(text) {
  // Convert **bold** to <strong>, *italic*, newlines to <br>
  return escapeHTML(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

function scrollChat(chat) {
  if (chat) chat.scrollTop = chat.scrollHeight;
}

// ══════════════════════════════════════════════════════════════
// USER PROFILE
// ══════════════════════════════════════════════════════════════
async function saveProfile() {
  const data = {
    name:          val('p-name'),
    age:           val('p-age'),
    gender:        val('p-gender'),
    weight_kg:     val('p-weight'),
    height_cm:     val('p-height'),
    diabetes_type: val('p-dtype'),
    glucose_level: val('p-glucose'),
    hba1c:         val('p-hba1c'),
    hypertension:  document.getElementById('p-hyp')?.checked ? 1 : 0,
    heart_disease: document.getElementById('p-hd')?.checked  ? 1 : 0,
    activity_level:val('p-activity'),
    dietary_pref:  val('p-diet'),
    region:        val('p-region'),
  };
  try {
    const d = await post('/save_profile', data);
    const el = document.getElementById('profile-saved-msg');
    if (el) { el.style.display='flex'; setTimeout(()=>el.style.display='none', 3000); }
    if (d.bmi) setText('p-bmi-display', `BMI: ${d.bmi}`);
  } catch (e) {
    showAlert('Could not save profile.', 'danger');
  }
}

// ══════════════════════════════════════════════════════════════
// FOOD LOG
// ══════════════════════════════════════════════════════════════
async function loadFoodLog() {
  try {
    const d  = await (await fetch('/food_log')).json();
    const el = document.getElementById('food-log-container');
    if (!el) return;

    if (!d.logs || !d.logs.length) {
      el.innerHTML = '<p class="text-muted text-sm">No foods logged yet. Analyze foods to start logging.</p>';
      return;
    }

    const rows = d.logs.map(r => `
      <tr>
        <td class="fw-bold">${r.food_name.replace(/_/g,' ')}</td>
        <td>${r.grams}g</td>
        <td>${r.calories}</td>
        <td>${r.carbs}g</td>
        <td>${r.glucose_imp}g</td>
        <td class="text-xs text-muted">${r.logged_at?.slice(0,16)||''}</td>
      </tr>`).join('');

    el.innerHTML = `
      <div class="info-box info" style="margin-bottom:1rem">
        Today's total: <strong>${d.total_calories} kcal</strong> |
        Carbs: <strong>${d.total_carbs}g</strong> |
        Glucose Impact: <strong>${d.total_glucose_impact}g</strong>
      </div>
      <table class="log-table">
        <thead><tr>
          <th>Food</th><th>Portion</th><th>Calories</th>
          <th>Carbs</th><th>Glucose</th><th>Time</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (e) { console.warn('Food log error:', e); }
}

// ══════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════
async function post(url, data) {
  const resp = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(()=>({error:`HTTP ${resp.status}`}));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

function val(id) {
  const el = document.getElementById(id);
  return el ? el.value : '';
}
function setVal(id, v) {
  const el = document.getElementById(id);
  if (el) el.value = v;
}
function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}
function setHTML(id, v) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = v;
}
function show(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'block';
}
function hide(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}
function showLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('show');
}
function hideLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('show');
}
function showAlert(msg, type='info') {
  const map = {info:'info',warning:'warning',danger:'danger',success:'success'};
  const div = document.createElement('div');
  div.className = `info-box ${map[type]||'info'}`;
  div.style.cssText = 'position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;max-width:360px;box-shadow:0 4px 20px rgba(0,0,0,.15)';
  div.textContent = msg;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3500);
}
function escapeHTML(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
async function fileToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload  = e => res(e.target.result.split(',')[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}
