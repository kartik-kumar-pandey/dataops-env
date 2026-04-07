import copy
from .models import Observation, Action

class DataOpsEnv:

    def __init__(self, task="hard"):
        self.task = task
        self.max_steps = 10
        self.reset()

    def reset(self):
        if self.task == "easy":
            self.data = [
                {"name": "", "age": "10", "email": "ravi@fixed.com"},
                {"name": "Amit", "age": "20", "email": "amit@fixed.com"}
            ]
        elif self.task == "medium":
            self.data = [
                {"name": "", "age": "twenty", "email": "ravi@fixed.com"},
                {"name": "Amit", "age": "20", "email": "amit@fixed.com"}
            ]
        else: # hard
            self.data = [
                {"name": "", "age": None, "email": "bademail"},
                {"name": "Ravi", "age": "-5", "email": "ravi@mail"},
                {"name": "Amit", "age": "twenty", "email": "amit@"}
            ]
        self.history = []
        self.step_count = 0
        return self.state()
        
    def state(self):
        return self._obs()

    def step(self, action: Action):
        self.step_count += 1
        reward = 0.0
        done = False
        error = None

        tool = action.action_type
        self.history.append(tool)

        try:
            if tool == "detect_missing":
                reward += 0.2

            elif tool == "fix_missing":
                for r in self.data:
                    if not r["name"]:
                        r["name"] = "Unknown"
                        reward += 0.2

            elif tool == "fix_outliers":
                for r in self.data:
                    if not str(r["age"]).isdigit() or int(r["age"]) < 0:
                        r["age"] = "0"
                        reward += 0.2

            elif tool == "standardize_format":
                for r in self.data:
                    if "@" not in r["email"]:
                        r["email"] += "@fixed.com"
                        reward += 0.2

            elif tool == "validate_schema":
                if self._is_clean():
                    reward += 0.5
                else:
                    reward -= 0.2

            elif tool == "generate_report":
                if self._is_clean():
                    reward += 1.0
                    done = True
                else:
                    reward -= 0.3

        except Exception as e:
            error = str(e)
            reward -= 0.3

        if self.step_count >= self.max_steps:
            done = True

        return self.state(), reward, done, {"error": error}

    def _obs(self):
        return Observation(
            data=self.data[:2],  # partial visibility
            errors=self._errors(),
            step_count=self.step_count
        )

    def _errors(self):
        errs = []
        for i, r in enumerate(self.data):
            if not r["name"]:
                errs.append(f"Missing name {i}")
            if not str(r["age"]).isdigit() or int(r["age"]) < 0:
                errs.append(f"Invalid age {i}")
            if "@" not in r["email"]:
                errs.append(f"Invalid email {i}")
        return errs

    def _is_clean(self):
        return len(self._errors()) == 0
