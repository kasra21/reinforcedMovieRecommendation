from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import tester

app = FastAPI()

class RecommendationInputs(BaseModel):
    genre: str
    country_origin: str
    cost: int
    main_actor: str
    object_id: Optional[int] = None

class ReinforceInputs(BaseModel):
    object_id: int
    like: int

@app.get("/")
def read_root():
    return {"Message": "Hello Welcome to our Recommender!"}


@app.post("/recommend")
def get_recommendation(rec_inputs: RecommendationInputs):
    recommendation = tester.create_recommendation(rec_inputs)
    print(recommendation)
    return recommendation

@app.post("/reinforce")
def get_reward(reinforce_inputs: ReinforceInputs):
    reward_object = tester.get_reward(reinforce_inputs)
    return reward_object

