from pydantic import BaseModel
from typing import List, Dict, Any

class Observation(BaseModel):
    data: List[Dict[str, Any]]
    errors: List[str]
    step_count: int


class Action(BaseModel):
    action_type: str
    params: Dict[str, Any]
