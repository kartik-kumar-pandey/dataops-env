"""
FastAPI application for the DataOps OpenEnv environment.

Endpoints:
  GET  /            — HTML DataOps Dashboard UI
  GET  /health      — liveness probe
  POST /reset       — reset environment, returns Observation
  POST /step        — step with an Action, returns (Observation, Reward, done, info)
  GET  /state       — current environment state
  GET  /tasks       — list available tasks
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from my_env.env import DataOpsEnv
from my_env.models import Action

app = FastAPI(
    title="DataOps — OpenEnv",
    description="A real-world OpenEnv environment where AI agents clean and validate data pipelines.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Multi-Session Registry
# ---------------------------------------------------------------------------
_envs: Dict[str, DataOpsEnv] = {}

TASKS = {
    "easy": {"id": "easy", "name": "Level 1: Easy Data", "difficulty": "easy", "desc": "Fix missing values only."},
    "medium": {"id": "medium", "name": "Level 2: Medium Data", "difficulty": "medium", "desc": "Fix missing values and invalid formats."},
    "hard": {"id": "hard", "name": "Level 3: Hard Data", "difficulty": "hard", "desc": "Full pipeline with schema validation and report generation."}
}

def _get_env(task_id: str) -> DataOpsEnv:
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Use: {list(TASKS.keys())}")
    if task_id not in _envs:
        _envs[task_id] = DataOpsEnv(task=task_id)
    return _envs[task_id]

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "hard"

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        if v is None:
            return cls()
        return cls(**v) if isinstance(v, dict) else v

class StepRequest(BaseModel):
    task_id: str = "hard"
    action_type: str
    params: Dict[str, Any] = {}

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "environment": "dataops", "version": "1.0.0"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASKS.values())}

@app.post("/reset", response_model=Dict[str, Any])
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    env = _get_env(req.task_id)
    obs = env.reset()
    return obs.dict() if hasattr(obs, 'dict') else obs

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    action = Action(action_type=req.action_type, params=req.params)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    
    obs_dict = obs.dict() if hasattr(obs, 'dict') else obs
    return StepResponse(
        observation=obs_dict,
        reward=reward,
        done=done,
        info=info,
    )

@app.get("/state")
def state(task_id: str = Query("hard")):
    env = _get_env(task_id)
    obs = env.state()
    return obs.dict() if hasattr(obs, 'dict') else obs

# ---------------------------------------------------------------------------
# DataOps Dashboard HTML UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def playground():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DataOps Dashboard — OpenEnv</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d2e; --border: #2a2d3e;
    --green: #00d68f; --red: #ff6b6b; --yellow: #ffd166;
    --blue: #4dabf7; --text: #e8eaf6; --muted: #888;
    --code-bg: #12151f;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 700; color: var(--blue); }
  header .badge { background: var(--border); color: var(--muted); font-size: 0.75rem; padding: 2px 8px; border-radius: 99px; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 0; height: calc(100vh - 60px); }
  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.5rem; }
  .main { display: flex; flex-direction: column; gap: 0; overflow: hidden; }
  
  .task-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; cursor: pointer; transition: border-color 0.2s; }
  .task-card:hover, .task-card.active { border-color: var(--blue); }
  .task-card .difficulty { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
  .easy { color: var(--green); } .medium { color: var(--yellow); } .hard { color: var(--red); }
  .task-card h3 { font-size: 0.9rem; margin: 0.4rem 0; }
  .task-card p { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
  
  .req-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 1.5rem; }
  .req-bar label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); display: block; margin-bottom: 0.4rem; }
  .req-text { font-size: 0.85rem; line-height: 1.6; color: var(--green); }
  
  .data-row { flex: 1; display: grid; grid-template-columns: 2fr 1fr; overflow: hidden; }
  .pane { display: flex; flex-direction: column; border-right: 1px solid var(--border); overflow: hidden; }
  .pane-header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0.6rem 1rem; font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; display: flex; justify-content: space-between; align-items: center; }
  
  .data-pane, .error-pane { padding: 1rem; background: var(--code-bg); overflow-y: auto; }
  
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { background: #1a1d2e; padding: 0.6rem; text-align: left; color: #b0b8d8; border: 1px solid var(--border); }
  td { padding: 0.6rem; border: 1px solid var(--border); color: #c8d6ff; }
  .error-list { list-style: none; display: flex; flex-direction: column; gap: 0.5rem; }
  .error-list li { background: #3a1515; border-left: 4px solid var(--red); padding: 0.8rem; font-family: monospace; font-size: 0.8rem; border-radius: 4px; }
  .success-banner { background: #123223; color: var(--green); border-radius: 8px; padding: 1.5rem; font-size: 1.2rem; text-align: center; border: 1px solid var(--green); margin-top: 1rem;}
  
  .action-panel { background: var(--surface); border-top: 1px solid var(--border); padding: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
  .action-btn { background: #2a2d3e; color: var(--text); padding: 0.8rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; transition: background 0.2s, transform 0.1s; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.3rem;}
  .action-btn:hover { background: var(--blue); }
  .action-btn:active { transform: scale(0.98); }
  
  .bottom-bar { background: #0f1117; border-top: 1px solid var(--border); padding: 0.75rem 1.5rem; display: flex; align-items: center; justify-content: space-between; }
  .reset-btn { background: var(--border); color: var(--text); padding: 0.5rem 1.5rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; margin-right: 1rem;}
  
  .score-bar { display: flex; align-items: center; gap: 1.5rem; }
  .stat-block { display: flex; flex-direction: column; align-items: center; }
  .stat-label { font-size: 0.65rem; text-transform: uppercase; color: var(--muted); }
  .stat-val { font-size: 1.2rem; font-weight: 700; color: var(--blue); }
  .reward-gained { color: var(--green); animation: fadeOutUp 1.5s forwards; font-weight: bold; position: absolute; margin-top: -20px;}
  
  @keyframes fadeOutUp {
    0% { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-20px); }
  }
</style>
</head>
<body>
<header>
  <h1>📊 DataOps Environment</h1>
  <span class="badge">OpenEnv</span>
  <span class="badge" style="color:var(--blue)">Data Cleaning</span>
</header>
<div class="layout">
  <div class="sidebar">
    <div>
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:0.8rem">Pipeline Tasks</div>
      <div id="task-list" style="display:flex;flex-direction:column;gap:0.6rem"></div>
    </div>
  </div>
  
  <div class="main">
    <div class="req-bar">
      <label>Current Status</label>
      <div class="req-text" id="req-text">Select a task to begin DataOps pipeline.</div>
    </div>
    
    <div class="data-row">
      <div class="pane">
        <div class="pane-header"><span>Observed Data (Partial Visibility)</span><span id="step-badge">Step 0/10</span></div>
        <div class="data-pane" id="data-pane">Loading table...</div>
      </div>
      <div class="pane">
        <div class="pane-header"><span>Pipeline Diagnostics</span></div>
        <div class="error-pane" id="error-pane"><p style="color:var(--muted)">No errors detected.</p></div>
      </div>
    </div>
    
    <div class="action-panel">
      <button class="action-btn" onclick="takeAction('detect_missing')"><span>🔍</span> Detect Missing</button>
      <button class="action-btn" onclick="takeAction('fix_missing')"><span>🔧</span> Fix Missing</button>
      <button class="action-btn" onclick="takeAction('fix_outliers')"><span>📈</span> Threshold Outliers</button>
      <button class="action-btn" onclick="takeAction('standardize_format')"><span>📐</span> Standardize Format</button>
      <button class="action-btn" onclick="takeAction('validate_schema')"><span>✔️</span> Validate Schema</button>
      <button class="action-btn" onclick="takeAction('generate_report')"><span>📄</span> Generate Final Report</button>
    </div>
    
    <div class="bottom-bar">
      <button class="reset-btn" onclick="resetEnv()">↺ Reset Pipeline</button>
      <div class="score-bar">
        <div class="stat-block">
          <span class="stat-label">Total Reward</span>
          <span class="stat-val" id="reward-display">0.00</span>
        </div>
        <div class="stat-block">
          <span class="stat-label">Errors Remaining</span>
          <span class="stat-val" style="color:var(--red)" id="error-count">0</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let currentTask = 'hard';
let tasks = [];
let totalReward = 0.0;

async function loadTasks() {
  const r = await fetch('/tasks');
  const d = await r.json();
  tasks = d.tasks;
  const el = document.getElementById('task-list');
  el.innerHTML = '';
  tasks.forEach(t => {
    const card = document.createElement('div');
    card.className = 'task-card' + (t.id === currentTask ? ' active' : '');
    card.innerHTML = `<div class="difficulty ${t.difficulty}">${t.difficulty}</div>
      <h3>${t.name}</h3>
      <p>${t.desc}</p>`;
    card.onclick = () => selectTask(t.id);
    el.appendChild(card);
  });
}

async function selectTask(taskId) {
  currentTask = taskId;
  totalReward = 0.0;
  document.querySelectorAll('.task-card').forEach((c,i) => {
    c.classList.toggle('active', tasks[i].id === taskId);
  });
  const r = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});
  const obs = await r.json();
  updateUI(obs);
}

function updateUI(obs, rewardGain = 0, done = false) {
  // Update Steps
  document.getElementById('step-badge').textContent = `Step ${obs.step_count}/10`;
  
  // Render Table
  const dp = document.getElementById('data-pane');
  if (obs.data && obs.data.length > 0) {
    const keys = Object.keys(obs.data[0]);
    let tHtml = `<table><thead><tr>`;
    keys.forEach(k => tHtml += `<th>${k.toUpperCase()}</th>`);
    tHtml += `</tr></thead><tbody>`;
    obs.data.forEach(row => {
      tHtml += `<tr>`;
      keys.forEach(k => {
        let val = row[k];
        if(val === null || val === "") val = '<span style="color:var(--red)">[NULL]</span>';
        tHtml += `<td>${val}</td>`;
      });
      tHtml += `</tr>`;
    });
    tHtml += `</tbody></table>`;
    dp.innerHTML = tHtml;
  } else {
    dp.innerHTML = '<p style="color:var(--muted)">No data observed.</p>';
  }

  // Render Errors
  const ep = document.getElementById('error-pane');
  document.getElementById('error-count').textContent = obs.errors.length;
  if (obs.errors.length > 0) {
    let eHtml = `<ul class="error-list">`;
    obs.errors.forEach(e => eHtml += `<li>🚨 ${e}</li>`);
    eHtml += `</ul>`;
    ep.innerHTML = eHtml;
    document.getElementById('error-count').style.color = 'var(--red)';
    document.getElementById('req-text').textContent = "Pipeline needs cleaning. Errors detected in schema.";
  } else {
    ep.innerHTML = '<p style="color:var(--green)">✅ Data is completely clean.</p>';
    document.getElementById('error-count').style.color = 'var(--green)';
    document.getElementById('req-text').textContent = "Pipeline data is healthy. Ready to generate report.";
  }
  
  // Animations and Done States
  if (rewardGain !== 0) {
    const floatText = document.createElement('div');
    floatText.className = 'reward-gained';
    floatText.textContent = (rewardGain > 0 ? '+' : '') + rewardGain.toFixed(2);
    document.getElementById('reward-display').parentElement.appendChild(floatText);
    setTimeout(() => floatText.remove(), 1500);
  }
  
  totalReward += rewardGain;
  document.getElementById('reward-display').textContent = totalReward.toFixed(2);
  
  if (done) {
    dp.innerHTML = `<div class="success-banner">🎉 DataOps Pipeline successfully completed! <br>Final Agent Reward: ${totalReward.toFixed(2)}</div>` + dp.innerHTML;
  }
}

async function takeAction(actionType) {
  const r = await fetch('/step', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task_id:currentTask, action_type: actionType, params: {}})
  });
  const d = await r.json();
  updateUI(d.observation, d.reward, d.done);
}

async function resetEnv() {
  await selectTask(currentTask);
}

// Init
(async () => {
  await loadTasks();
  await selectTask('hard');
})();
</script>
</body>
</html>"""

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
