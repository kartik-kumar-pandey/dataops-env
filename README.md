---
title: DataOps Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Autonomous DataOps Agent Environment

## 1. Problem
Data cleaning is critical in real-world pipelines. Dealing with missing data, invalid schema formats, and broken dependencies can often cause failures in downstream processing and analytics.

## 2. Solution
An RL-based DataOps environment that simulates an end-to-end data pipeline.

## 3. Features
- **Tool-based actions**: Use tools like `detect_missing`, `fix_missing`, `validate_schema` instead of simple one-off actions.
- **Hidden state**: Agent does not see full truths, requiring logical deduction over observations.
- **Multi-step reasoning**: Agent must follow the correct processing sequence `detect → fix → validate → report`.

## 4. Tasks
- **Easy**: Fix missing values only
- **Medium**: Fix missing + invalid formats
- **Hard**: Full pipeline with validation + report

## 5. Reward Design
The environment shapes rewards dense and strict:
- Correct tool usage: +0.2
- Fix issue: +0.3
- Skip step / Invalid action: Penalties up to -0.3
- Final success: +1.0

## 6. Results
Run the test inference script to verify correct behavior:
```bash
python inference.py
```
