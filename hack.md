# 🏆 Autonomous DataOps Agent Environment

An advanced, production-grade RL environment built to OpenEnv specifications, turning complex Data Engineering and validation pipelines into a solvable problem space for autonomous AI agents.

## 🔥 Key Novelty
While most starter environments focus on toy tasks, this submission solves **Real-World Business Workflows**—specifically, resilient ETL / Data Operational error resolution. 

Instead of full oracle visibility, the Agent is tested against a **Hidden State Pipeline** where it only receives localized feedback (`obs.errors`), forcing the AI to autonomously reason, sequentially choose abstract functions (Tools), and execute them efficiently to secure a 1.0 grade. 

---

## 🏗️ Core Features Matrix

### 1. 100% OpenEnv Spec Compliant
- **Strict Architecture**: Fully implements expected `reset()`, `step()`, and the critical `state()` endpoint mapped against isolated typed data schemas (via strict Pydantic `models.py` definitions).
- **Format-Perfect Stdout**: The `inference.py` adheres to the absolute exact boolean/string schemas required by the internal bash OpenEnv validation validators (`[START]`, `[STEP]`, `[END]`, `error=null`, `done=true`).

### 2. Multi-Task Hidden State Architecture
The environment supports three wildly different baseline JSON datasets (Easy, Medium, Hard). The Agent is completely blind to the underlying arrays. To solve them, it relies purely on reading the partial data Observation space state and picking a tool. 

If the agent invokes invalid sequences or forces actions unreliably, it accumulates negative rewards and loses points. 

### 3. Dynamic Meta/OpenAI Client Engine
We embedded the official `openai` Python client directly into the core `inference.py` script. The Agent configures `API_BASE_URL` and `MODEL_NAME` to natively support HuggingFace Router endpoints **and** third-party vendor APIs like NVIDIA natively without codebase modifications.

### 4. Robust Execution Design
The inference pipeline includes a fallback execution strategy to ensure reproducibility in environments where external API calls may fail (e.g., automated judging containers with restricted egress).

This fallback uses a deterministic baseline policy that executes a predefined tool sequence. While it ensures the script completes successfully and avoids runtime crashes, it does not replace agent reasoning and is solely utilized for absolute system robustness.

---

## 🧠 Why This Environment is Challenging

- **Partial observability** (agent does not see full dataset)
- **Requires correct tool sequencing**
- **Penalizes premature validation**
- **Multiple interacting data errors** (missing, format, outliers)

This ensures the environment meaningfully evaluates true structural reasoning ability rather than simple pattern matching.

---

## 📐 Design Principles

- **Deterministic transitions** (no randomness)
- **Reproducible evaluation**
- **Dense reward shaping** for targeted learning signals
- **Clear episode termination conditions**

---

## 📥 Observation Space

The agent evaluates the environment state through a partial, hidden-state observation structure:
- **Partial Dataset**: A visibly gated subset of the dataframe rows.
- **Detected Errors List**: Explicit array of formatting/outlier strings currently tracked.
- **Step Count**: Integer tracking usage against execution limits.

---

## 🎯 Action Space

Instead of arbitrary button clicks, the Agent selects from high-level data functions, mimicking an enterprise ETL operator:

1. `detect_missing`
2. `fix_missing`
3. `fix_outliers`
4. `standardize_format`
5. `validate_schema`
6. `generate_report`

**Dense Rewards:** 
Correct choices yield `+0.2 → +0.6`. Running incorrect schemas, attempting validations early, or repeating failures deduct points severely via strict Python graders targeting optimal sequence length limits.

---

## 🚀 How to Execute & Verify

**(A) The Grading Baseline Run**
Just execute the baseline locally on Hard mode:
```powershell
$env:MY_ENV_TASK="hard"
python inference.py
```

### 📊 Example Trajectory
```text
[START] task=hard env=dataops model=baseline
[STEP] step=1 action=detect_missing reward=0.20 done=false error=null
[STEP] step=2 action=fix_missing reward=0.20 done=false error=null
[STEP] step=3 action=fix_outliers reward=0.60 done=false error=null
[STEP] step=4 action=standardize_format reward=0.20 done=false error=null
[STEP] step=5 action=validate_schema reward=0.50 done=false error=null
[STEP] step=6 action=generate_report reward=1.00 done=true error=null
[END] success=true steps=6 score=1.00 rewards=0.20,0.20,0.60,0.20,0.50,1.00
```

**(B) The Dynamic AI "Skip" Solve (Autonomous LLM Agent)**
Set up the environment using your LLM access key on the "Easy" mode:
```powershell
$env:MY_ENV_TASK="easy"
$env:API_BASE_URL="https://integrate.api.nvidia.com/v1"
$env:MODEL_NAME="meta/llama-3.1-70b-instruct"
$env:HF_TOKEN="<your_nv_token>"
python inference.py
```
