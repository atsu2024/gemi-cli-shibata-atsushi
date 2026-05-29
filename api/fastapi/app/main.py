from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str
    status: Literal["ok"]
    timestamp: datetime


class SimulationRequest(BaseModel):
    name: str = Field(..., min_length=1, examples=["lorenz"])
    steps: int = Field(100, ge=1, le=100000)
    initial_value: float = Field(1.0)


class SimulationResponse(BaseModel):
    name: str
    steps: int
    initial_value: float
    result: float


app = FastAPI(
    title="Scientific FastAPI Service",
    version="1.0.0",
    description="FastAPI API for high-precision scientific service workflows.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="fastapi",
        status="ok",
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest) -> SimulationResponse:
    if payload.name.lower() not in {"lorenz", "dnn", "precision"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported simulation. Use one of: lorenz, dnn, precision.",
        )

    value = payload.initial_value
    for step in range(payload.steps):
        value += (step + 1) / (payload.steps + 1)
        value *= 0.999

    return SimulationResponse(
        name=payload.name,
        steps=payload.steps,
        initial_value=payload.initial_value,
        result=value,
    )
