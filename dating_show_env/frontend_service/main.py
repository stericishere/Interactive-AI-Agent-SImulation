#!/usr/bin/env python3
"""
Dating Show Frontend Service
FastAPI-based dedicated frontend service for the generative agents dating show
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import uvicorn
import json
import os
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime

from core.models import SimulationState, AgentState, EnvironmentState
from core.websocket_manager import WebSocketManager
from core.simulation_bridge import SimulationBridge
from core.config import Settings

settings = Settings()
app = FastAPI(title="Dating Show Frontend Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

websocket_manager = WebSocketManager()
simulation_bridge = SimulationBridge()

# Use our Jinja2-compatible templates
templates = Jinja2Templates(directory="templates")
# Mount original Django static files
app.mount("/static", StaticFiles(directory="../../environment/frontend_server/static_dirs"), name="static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Landing page"""
    try:
        return templates.TemplateResponse(
            "landing/landing.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error loading landing page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load landing page")

@app.get("/simulator_home", response_class=HTMLResponse)
async def home(request: Request):
    """Main simulation view using original Django template"""
    try:
        sim_state = await simulation_bridge.get_current_simulation_state()
        
        if not sim_state or not sim_state.agents:
            # No simulation running, show error page
            return templates.TemplateResponse(
                "home/error_start_backend.html",
                {"request": request}
            )
        
        # Convert agents to Django template format
        persona_names = [[agent.name, agent.name.replace(" ", "_")] for agent in sim_state.agents]
        persona_init_pos = [[agent.name, agent.position.x, agent.position.y] for agent in sim_state.agents]
        
        return templates.TemplateResponse(
            "home/home.html",
            {
                "request": request,
                "sim_code": sim_state.sim_code,
                "step": sim_state.current_step,
                "persona_names": persona_names,
                "persona_init_pos": persona_init_pos,
                "mode": "simulate"
            }
        )
    except Exception as e:
        logger.error(f"Error loading simulation: {e}")
        return templates.TemplateResponse(
            "home/error_start_backend.html",
            {"request": request}
        )

@app.get("/api/simulation/state")
async def get_simulation_state():
    """Get current simulation state"""
    return await simulation_bridge.get_current_simulation_state()

@app.get("/api/agents")
async def get_agents():
    """Get all agent states"""
    return await simulation_bridge.get_all_agents()

@app.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get specific agent state"""
    agent = await simulation_bridge.get_agent_state(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await websocket_manager.handle_message(websocket, message)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

@app.post("/api/simulation/step")
async def advance_simulation():
    """Advance simulation by one step"""
    try:
        result = await simulation_bridge.advance_simulation()
        await websocket_manager.broadcast_simulation_update(result)
        return result
    except Exception as e:
        logger.error(f"Error advancing simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to advance simulation")

# Django-compatible endpoints for Phaser.js integration
@app.post("/process_environment/")
async def process_environment(request: Request):
    """Process environment data from Phaser frontend"""
    try:
        data = await request.json()
        step = data.get("step")
        sim_code = data.get("sim_code")
        environment = data.get("environment")
        
        # Store environment data (in real implementation, this would save to storage)
        logger.info(f"Processed environment for step {step}, sim_code {sim_code}")
        
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Error processing environment: {e}")
        raise HTTPException(status_code=500, detail="Failed to process environment")

@app.post("/update_environment/")
async def update_environment(request: Request):
    """Get movement updates for Phaser frontend"""
    try:
        data = await request.json()
        step = data.get("step")
        sim_code = data.get("sim_code")
        
        # Get current simulation state
        sim_state = await simulation_bridge.get_current_simulation_state()
        
        if not sim_state:
            return {"<step>": -1}
        
        # Build movement data in Django format
        movement_data = {"<step>": step}
        
        for agent in sim_state.agents:
            agent_name = agent.name
            movement_data[agent_name] = {
                "movement": [agent.position.x, agent.position.y],
                "pronunciatio": agent.current_action or "idle",
                "description": f"{agent.name} is {agent.current_action or 'idle'} at {agent.current_location or 'unknown location'}",
                "chat": getattr(agent, 'current_chat', ''),
                "scratch": {
                    "curr_tile": [agent.current_location or "unknown"],
                    "daily_plan_req": agent.current_action or "idle"
                }
            }
        
        return movement_data
        
    except Exception as e:
        logger.error(f"Error updating environment: {e}")
        return {"<step>": -1}

# Additional Django-compatible routes
@app.get("/demo/{sim_code}/{step}/{play_speed}/", response_class=HTMLResponse)
async def demo(request: Request, sim_code: str, step: int, play_speed: str = "2"):
    """Demo replay view"""
    try:
        return templates.TemplateResponse(
            "demo/demo.html",
            {
                "request": request,
                "sim_code": sim_code,
                "step": step,
                "play_speed": play_speed,
                "mode": "demo"
            }
        )
    except Exception as e:
        logger.error(f"Error loading demo: {e}")
        raise HTTPException(status_code=500, detail="Failed to load demo")

@app.get("/replay/{sim_code}/{step}/", response_class=HTMLResponse)
async def replay(request: Request, sim_code: str, step: int):
    """Replay view"""
    try:
        sim_state = await simulation_bridge.get_current_simulation_state()
        
        if not sim_state:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        persona_names = [[agent.name, agent.name.replace(" ", "_")] for agent in sim_state.agents]
        persona_init_pos = [[agent.name, agent.position.x, agent.position.y] for agent in sim_state.agents]
        
        return templates.TemplateResponse(
            "home/home.html",
            {
                "request": request,
                "sim_code": sim_code,
                "step": step,
                "persona_names": persona_names,
                "persona_init_pos": persona_init_pos,
                "mode": "replay"
            }
        )
    except Exception as e:
        logger.error(f"Error loading replay: {e}")
        raise HTTPException(status_code=500, detail="Failed to load replay")

@app.get("/replay_persona_state/{sim_code}/{step}/{persona_name}/", response_class=HTMLResponse)
async def replay_persona_state(request: Request, sim_code: str, step: int, persona_name: str):
    """Individual persona state view"""
    try:
        sim_state = await simulation_bridge.get_current_simulation_state()
        
        if not sim_state:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        # Find the agent
        agent = None
        for a in sim_state.agents:
            if a.name.replace(" ", "_") == persona_name:
                agent = a
                break
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return templates.TemplateResponse(
            "persona_state/persona_state.html",
            {
                "request": request,
                "sim_code": sim_code,
                "step": step,
                "persona_name": agent.name,
                "persona_name_underscore": persona_name,
                "scratch": {
                    "curr_tile": [agent.current_location or "unknown"],
                    "daily_plan_req": agent.current_action or "idle"
                },
                "spatial": {},
                "a_mem_event": [],
                "a_mem_chat": agent.dialogue_history,
                "a_mem_thought": []
            }
        )
    except Exception as e:
        logger.error(f"Error loading persona state: {e}")
        raise HTTPException(status_code=500, detail="Failed to load persona state")

@app.get("/path_tester/", response_class=HTMLResponse)
async def path_tester(request: Request):
    """Path tester utility"""
    try:
        return templates.TemplateResponse(
            "path_tester/path_tester.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error loading path tester: {e}")
        raise HTTPException(status_code=500, detail="Failed to load path tester")

@app.post("/path_tester_update/")
async def path_tester_update(request: Request):
    """Path tester update endpoint"""
    try:
        data = await request.json()
        camera = data.get("camera")
        
        # In real implementation, this would save to temp storage
        logger.info(f"Path tester update: {camera}")
        
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Error updating path tester: {e}")
        raise HTTPException(status_code=500, detail="Failed to update path tester")

# Dating Show API endpoints that the backend expects
@app.post("/dating_show/api/agents/{agent_id}/state/")
async def update_agent_state(agent_id: str, request: Request):
    """Receive agent state updates from the dating show backend"""
    try:
        agent_data = await request.json()
        
        logger.info(f"Received agent state update for {agent_id}: {agent_data.get('name', 'Unknown')}")
        
        # Broadcast the agent update via WebSocket
        await websocket_manager.broadcast_agent_update(agent_id, agent_data)
        
        return {"status": "received", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"Error updating agent state for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update agent state")

@app.get("/dating_show/api/agents/{agent_id}/state/")
async def get_agent_state_by_id(agent_id: str):
    """Get specific agent state by ID"""
    try:
        agents = await simulation_bridge.get_all_agents()
        
        # Find agent by ID or name
        for agent in agents:
            if (hasattr(agent, 'agent_id') and agent.agent_id == agent_id) or agent.name == agent_id:
                return {
                    "agent_id": agent_id,
                    "name": agent.name,
                    "role": agent.role,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "current_action": agent.current_action,
                    "current_location": agent.current_location,
                    "emotional_state": agent.emotional_state,
                    "relationship_scores": agent.relationship_scores,
                    "last_updated": agent.last_updated.isoformat()
                }
        
        raise HTTPException(status_code=404, detail="Agent not found")
        
    except Exception as e:
        logger.error(f"Error getting agent state for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent state")

@app.post("/dating_show/api/simulation/sync/")
async def sync_simulation_data(request: Request):
    """Receive bulk simulation data from the dating show backend"""
    try:
        sync_data = await request.json()
        
        logger.info(f"Received simulation sync: {len(sync_data.get('agents', []))} agents")
        
        # Broadcast simulation update
        await websocket_manager.broadcast({
            "type": "simulation_sync",
            "data": sync_data,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "received", "agents_count": len(sync_data.get('agents', []))}
        
    except Exception as e:
        logger.error(f"Error syncing simulation data: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync simulation data")

@app.get("/dating_show/api/simulation/status/")
async def get_simulation_status():
    """Get current simulation status"""
    try:
        sim_state = await simulation_bridge.get_current_simulation_state()
        
        if not sim_state:
            return {"status": "no_simulation", "active": False}
        
        return {
            "status": "running",
            "active": True,
            "sim_code": sim_state.sim_code,
            "current_step": sim_state.current_step,
            "agents_count": len(sim_state.agents),
            "mode": sim_state.mode,
            "uptime": (datetime.now() - sim_state.start_time).total_seconds() if sim_state.start_time else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        return {"status": "error", "active": False, "error": str(e)}

@app.post("/dating_show/api/broadcast/update/")
async def broadcast_update(request: Request):
    """Receive broadcast updates from the dating show backend"""
    try:
        update_data = await request.json()
        
        logger.info(f"Received broadcast update: {update_data.get('type', 'unknown')} - {update_data.get('message', 'no message')}")
        
        # Broadcast to all connected WebSocket clients
        message = {
            "type": "broadcast_update",
            "data": update_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast(json.dumps(message))
        
        return {"status": "received", "broadcast_type": update_data.get('type', 'unknown')}
        
    except Exception as e:
        logger.error(f"Error handling broadcast update: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle broadcast update")

@app.post("/dating_show/api/broadcast/agent_update/")
async def broadcast_agent_update(request: Request):
    """Receive agent-specific broadcast updates"""
    try:
        agent_update = await request.json()
        
        agent_id = agent_update.get('agent_id', 'unknown')
        logger.info(f"Received agent broadcast update for {agent_id}")
        
        # Broadcast agent-specific update
        await websocket_manager.broadcast_agent_update(agent_id, agent_update)
        
        return {"status": "received", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"Error handling agent broadcast update: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle agent broadcast update")

@app.post("/dating_show/api/broadcast/simulation_event/")
async def broadcast_simulation_event(request: Request):
    """Receive simulation event broadcasts"""
    try:
        event_data = await request.json()
        
        event_type = event_data.get('event_type', 'unknown')
        logger.info(f"Received simulation event broadcast: {event_type}")
        
        # Broadcast simulation event
        message = {
            "type": "simulation_event",
            "event_type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast(json.dumps(message))
        
        return {"status": "received", "event_type": event_type}
        
    except Exception as e:
        logger.error(f"Error handling simulation event broadcast: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle simulation event broadcast")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Dating Show Frontend Service")
    await simulation_bridge.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Dating Show Frontend Service")
    await simulation_bridge.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )