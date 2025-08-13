"""
Pydantic data models for the Dating Show Frontend Service
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class AgentRole(str, Enum):
    """Dating show agent roles"""
    CONTESTANT = "contestant"
    HOST = "host"
    OBSERVER = "observer"


class SimulationMode(str, Enum):
    """Simulation operation modes"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class Position(BaseModel):
    """Agent position in the environment"""
    x: float
    y: float
    sector: Optional[str] = None


class AgentState(BaseModel):
    """Current state of a dating show agent"""
    name: str
    role: AgentRole
    position: Position
    current_action: Optional[str] = None
    current_location: Optional[str] = None
    emotional_state: Dict[str, float] = Field(default_factory=dict)
    relationship_scores: Dict[str, float] = Field(default_factory=dict)
    dialogue_history: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)


class EnvironmentState(BaseModel):
    """Current environment state"""
    locations: Dict[str, Any] = Field(default_factory=dict)
    active_events: List[str] = Field(default_factory=list)
    time_of_day: str = "morning"
    weather: str = "sunny"


class SimulationState(BaseModel):
    """Overall simulation state"""
    sim_code: str
    current_step: int
    mode: SimulationMode = SimulationMode.STOPPED
    agents: List[AgentState] = Field(default_factory=list)
    environment: EnvironmentState = Field(default_factory=EnvironmentState)
    start_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SimulationUpdate(BaseModel):
    """Simulation update message"""
    step: int
    agents_updated: List[str]
    environment_changed: bool = False
    events: List[str] = Field(default_factory=list)