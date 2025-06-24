from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import os
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome!"}

# Example column names with hyphens
# Assume your model expects these features:
FEATURE_NAMES = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours-per-week",
    "native-country",
]


# random_forest model
class random_forest:
    def predict(self, X: np.ndarray) -> List[int]:
        # random_forest logic: predict 1 if age > 30, else 0
        return [1 if x[0] > 30 else 0 for x in X]


model = random_forest()


# Pydantic model for request body
class CensusData(BaseModel):
    age: int
    workclass: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData) -> dict:
    # Convert input data to numpy array in the order expected by the model
    input_array = np.array(
        [
            [
                data.age,
                # For categorical vars, your real pipeline would encode them;
                1 if data.workclass == "State-gov" else 0,
                data.education_num,
                1 if data.marital_status == "Never-married" else 0,
                1 if data.occupation == "Adm-clerical" else 0,
                1 if data.relationship == "Not-in-family" else 0,
                1 if data.race == "White" else 0,
                1 if data.sex == "Male" else 0,
                data.hours_per_week,
                1 if data.native_country == "United-States" else 0,
            ]
        ]
    )

    prediction = model.predict(input_array)[0]

    return {"prediction": prediction}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)