from fastapi import FastAPI
from my_env.env import DataOpsEnv
from my_env.models import Action

app = FastAPI()
env = DataOpsEnv()

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/reset")
def reset_env():
    return env.reset()

@app.post("/step")
def step_env(action: dict):
    return env.step(Action(**action))
