"""
File: identity_persistence.py
Description: Professional Identity Persistence with LangGraph Store API for Task 3.3.
Handles professional identity storage, cross-session continuity, and identity evolution tracking.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import logging
import uuid
import statistics
import time

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager, SpecializationData
from ..memory_structures.store_integration import MemoryStoreIntegration, StoreNamespace


class IdentityDevelopmentPhase(Enum):
    """Phases of professional identity development."""
    EXPLORATION = "exploration"        # Agent discovering different roles
    COMMITMENT = "commitment"          # Agent committing to specific role
    SYNTHESIS = "synthesis"           # Agent integrating role with personal values
    MASTERY = "mastery"              # Agent achieving expertise in role
    RENEWAL = "renewal"              # Agent evolving or transitioning roles


class IdentityStrength(Enum):
    """Strength levels of professional identity."""
    WEAK = "weak"              # 0.0-0.3: Identity is forming or uncertain
    MODERATE = "moderate"      # 0.3-0.6: Identity is developing
    STRONG = "strong"          # 0.6-0.8: Identity is well-established
    VERY_STRONG = "very_strong"  # 0.8-1.0: Identity is deeply integrated


@dataclass
class IdentityMilestone:
    """Milestone in professional identity development."""
    milestone_id: str
    milestone_type: str  # 'role_discovery', 'skill_mastery', 'identity_crisis', 'role_transition'
    description: str
    achieved_at: datetime
    role_context: str
    impact_score: float  # 0.0-1.0, how much this impacted identity
    evidence_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentityEvolutionRecord:
    """Record of identity evolution over time."""
    record_id: str
    agent_id: str
    from_role: str
    to_role: str
    transition_period: timedelta
    catalyst_events: List[str]
    confidence_change: float
    skill_transfers: Dict[str, float]
    challenges_faced: List[str]
    support_received: List[str]
    recorded_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfessionalIdentity:
    """Complete professional identity data structure."""
    identity_id: str
    agent_id: str
    
    # Core Identity Components
    primary_role: str
    role_confidence: float
    role_commitment: float
    role_satisfaction: float
    
    # Identity Development
    development_phase: IdentityDevelopmentPhase
    identity_strength: IdentityStrength
    formation_date: datetime
    last_evolution_date: Optional[datetime]
    
    # Historical Development
    identity_history: List[Dict[str, Any]]
    milestone_history: List[IdentityMilestone]
    evolution_records: List[IdentityEvolutionRecord]
    
    # Identity Coherence
    value_alignment: float  # How well role aligns with personal values
    behavior_consistency: float  # How consistently agent acts in role
    social_validation: float  # How much others validate this identity
    internal_coherence: float  # How internally consistent the identity is
    
    # Performance and Growth
    role_performance_metrics: Dict[str, float]
    growth_trajectory: List[Tuple[datetime, float]]
    learning_patterns: Dict[str, Any]
    
    # Cross-Session Continuity
    session_consistency_score: float
    context_adaptation_ability: float
    memory_integration_quality: float
    
    # Metadata
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityPersistenceModule(LangGraphBaseModule):
    """
    Professional Identity Persistence Module for Task 3.3.
    Manages professional identity storage, cross-session continuity, and evolution tracking.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None,
                 store_integration: Optional[MemoryStoreIntegration] = None):
        """
        Initialize Identity Persistence Module.
        
        Args:
            state_manager: Enhanced agent state manager
            store_integration: LangGraph Store API integration
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.SLOW,  # Identity changes slowly
            priority=ModulePriority.MEDIUM,
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=5.0
        )
        
        super().__init__("identity_persistence", config, state_manager)
        
        self.store_integration = store_integration
        self.identity_cache: Dict[str, ProfessionalIdentity] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl = 600  # 10 minutes
        
        # Identity tracking settings
        self.identity_update_threshold = 0.05  # Minimum change to trigger update
        self.milestone_importance_threshold = 0.7
        self.evolution_tracking_days = 30
        self.coherence_calculation_interval = timedelta(hours=1)
        
        # Performance tracking
        self.persistence_stats = {
            "identity_updates": 0,
            "milestones_recorded": 0,
            "evolution_records": 0,
            "cross_session_continuity_score": 0.0,
            "avg_identity_coherence": 0.0,
            "last_coherence_calculation": None
        }
        
        # State dependencies
        self.add_state_dependency("specialization")
        
        self.logger = logging.getLogger("IdentityPersistence")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for identity persistence and evolution tracking.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with identity persistence results
        """
        start_time = time.time()
        
        try:
            agent_id = state.get("agent_id", "unknown")
            
            # Load or create professional identity
            identity = self._load_or_create_identity(state)
            
            # Update identity based on current state
            identity_changes = self._update_identity_from_state(identity, state)
            
            # Detect identity milestones
            milestones = self._detect_identity_milestones(identity, state)
            
            # Track identity evolution
            evolution_changes = self._track_identity_evolution(identity, state)
            
            # Calculate identity coherence
            coherence_metrics = self._calculate_identity_coherence(identity, state)
            
            # Update cross-session continuity
            continuity_updates = self._update_cross_session_continuity(identity, state)
            
            # Persist identity if significant changes occurred
            if identity_changes or milestones or evolution_changes:
                self._persist_identity(identity)
            
            # Update performance statistics
            self._update_performance_stats(identity, milestones, evolution_changes)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "professional_identity": identity
                },
                "output_data": {
                    "identity_status": {
                        "primary_role": identity.primary_role,
                        "development_phase": identity.development_phase.value,
                        "identity_strength": identity.identity_strength.value,
                        "role_confidence": identity.role_confidence
                    },
                    "identity_changes": {
                        "significant_changes": identity_changes,
                        "new_milestones": len(milestones),
                        "evolution_detected": bool(evolution_changes)
                    },
                    "coherence_metrics": coherence_metrics,
                    "continuity_quality": {
                        "session_consistency": identity.session_consistency_score,
                        "context_adaptation": identity.context_adaptation_ability,
                        "memory_integration": identity.memory_integration_quality
                    }
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "identity_coherence": identity.internal_coherence,
                    "development_progress": self._calculate_development_progress(identity)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in identity persistence: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _load_or_create_identity(self, state: EnhancedAgentState) -> ProfessionalIdentity:
        """Load existing identity or create new one."""
        agent_id = state.get("agent_id", "unknown")
        
        # Check cache first
        if agent_id in self.identity_cache:
            cache_key = f"identity_{agent_id}"
            if cache_key in self.cache_expiry and datetime.now() < self.cache_expiry[cache_key]:
                return self.identity_cache[agent_id]
        
        # Try to load from Store API
        existing_identity = self._load_identity_from_store(agent_id)
        if existing_identity:
            self._cache_identity(existing_identity)
            return existing_identity
        
        # Create new identity
        return self._create_new_identity(state)
    
    def _load_identity_from_store(self, agent_id: str) -> Optional[ProfessionalIdentity]:
        """Load identity from LangGraph Store API."""
        try:
            if not self.store_integration or not self.store_integration.store:
                return None
            
            # Note: This would be async in real implementation
            # For now, we'll simulate the structure
            identity_key = f"professional_identity_{agent_id}"
            
            # In real implementation, this would be:
            # identity_data = await self.store_integration.store.aget("professional_identities", identity_key)
            
            # Simulate loading from store for now
            return None
        
        except Exception as e:
            self.logger.error(f"Error loading identity from store for {agent_id}: {str(e)}")
            return None
    
    def _create_new_identity(self, state: EnhancedAgentState) -> ProfessionalIdentity:
        """Create new professional identity for agent."""
        agent_id = state.get("agent_id", "unknown")
        agent_name = state.get("name", "Unknown")
        specialization = state.get("specialization")
        
        current_role = "contestant"  # Default role
        if specialization and hasattr(specialization, 'current_role'):
            current_role = specialization.current_role
        
        identity = ProfessionalIdentity(
            identity_id=str(uuid.uuid4()),
            agent_id=agent_id,
            
            # Core Identity Components
            primary_role=current_role,
            role_confidence=0.5,  # Start with moderate confidence
            role_commitment=0.3,  # Start with low commitment
            role_satisfaction=0.5,
            
            # Identity Development
            development_phase=IdentityDevelopmentPhase.EXPLORATION,
            identity_strength=IdentityStrength.WEAK,
            formation_date=datetime.now(),
            last_evolution_date=None,
            
            # Historical Development
            identity_history=[{
                "event": "identity_creation",
                "role": current_role,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.5,
                "metadata": {"initial_creation": True}
            }],
            milestone_history=[],
            evolution_records=[],
            
            # Identity Coherence
            value_alignment=0.5,
            behavior_consistency=0.5,
            social_validation=0.5,
            internal_coherence=0.5,
            
            # Performance and Growth
            role_performance_metrics={},
            growth_trajectory=[(datetime.now(), 0.5)],
            learning_patterns={},
            
            # Cross-Session Continuity
            session_consistency_score=1.0,  # Start perfect, will adjust
            context_adaptation_ability=0.5,
            memory_integration_quality=0.5,
            
            # Metadata
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={"agent_name": agent_name, "initial_specialization": asdict(specialization) if specialization else {}}
        )
        
        self._cache_identity(identity)
        return identity
    
    def _update_identity_from_state(self, identity: ProfessionalIdentity, 
                                   state: EnhancedAgentState) -> bool:
        """Update identity based on current agent state."""
        changes_made = False
        specialization = state.get("specialization")
        
        if not specialization:
            return changes_made
        
        # Update role information
        if hasattr(specialization, 'current_role') and specialization.current_role != identity.primary_role:
            old_role = identity.primary_role
            identity.primary_role = specialization.current_role
            identity.last_evolution_date = datetime.now()
            
            # Record role change in history
            identity.identity_history.append({
                "event": "role_change",
                "from_role": old_role,
                "to_role": specialization.current_role,
                "timestamp": datetime.now().isoformat(),
                "confidence": getattr(specialization, 'role_consistency_score', 0.5)
            })
            
            changes_made = True
            self.logger.info(f"Role change detected: {old_role} -> {specialization.current_role}")
        
        # Update role confidence
        if hasattr(specialization, 'role_consistency_score'):
            new_confidence = specialization.role_consistency_score
            confidence_change = abs(new_confidence - identity.role_confidence)
            
            if confidence_change >= self.identity_update_threshold:
                identity.role_confidence = new_confidence
                changes_made = True
        
        # Update expertise level as role performance
        if hasattr(specialization, 'expertise_level'):
            identity.role_performance_metrics["expertise"] = specialization.expertise_level
            changes_made = True
        
        # Update skills data
        if hasattr(specialization, 'skills') and specialization.skills:
            identity.role_performance_metrics.update({
                f"skill_{skill}": level for skill, level in specialization.skills.items()
            })
            changes_made = True
        
        # Update development phase based on role confidence and commitment
        new_phase = self._determine_development_phase(identity)
        if new_phase != identity.development_phase:
            identity.development_phase = new_phase
            changes_made = True
        
        # Update identity strength
        new_strength = self._calculate_identity_strength(identity)
        if new_strength != identity.identity_strength:
            identity.identity_strength = new_strength
            changes_made = True
        
        if changes_made:
            identity.last_updated = datetime.now()
            # Add growth trajectory point
            identity.growth_trajectory.append((datetime.now(), identity.role_confidence))
            # Keep only last 50 points
            if len(identity.growth_trajectory) > 50:
                identity.growth_trajectory = identity.growth_trajectory[-50:]
        
        return changes_made
    
    def _determine_development_phase(self, identity: ProfessionalIdentity) -> IdentityDevelopmentPhase:
        """Determine current identity development phase."""
        confidence = identity.role_confidence
        commitment = identity.role_commitment
        days_since_formation = (datetime.now() - identity.formation_date).days
        
        # Exploration phase: Low confidence, exploring different roles
        if confidence < 0.4 or days_since_formation < 7:
            return IdentityDevelopmentPhase.EXPLORATION
        
        # Commitment phase: Building confidence in chosen role
        elif confidence < 0.7 and commitment < 0.7:
            return IdentityDevelopmentPhase.COMMITMENT
        
        # Synthesis phase: Integrating role with personal values
        elif confidence < 0.8 and identity.value_alignment < 0.8:
            return IdentityDevelopmentPhase.SYNTHESIS
        
        # Mastery phase: High competence and integration
        elif confidence >= 0.8 and commitment >= 0.7:
            return IdentityDevelopmentPhase.MASTERY
        
        # Renewal phase: Evolving or transitioning (detected by recent role changes)
        elif identity.last_evolution_date and (datetime.now() - identity.last_evolution_date).days < 14:
            return IdentityDevelopmentPhase.RENEWAL
        
        return identity.development_phase  # Keep current phase if no clear transition
    
    def _calculate_identity_strength(self, identity: ProfessionalIdentity) -> IdentityStrength:
        """Calculate overall identity strength."""
        # Weighted average of key components
        strength_score = (
            identity.role_confidence * 0.3 +
            identity.role_commitment * 0.2 +
            identity.behavior_consistency * 0.2 +
            identity.social_validation * 0.15 +
            identity.internal_coherence * 0.15
        )
        
        if strength_score >= 0.8:
            return IdentityStrength.VERY_STRONG
        elif strength_score >= 0.6:
            return IdentityStrength.STRONG
        elif strength_score >= 0.3:
            return IdentityStrength.MODERATE
        else:
            return IdentityStrength.WEAK
    
    def _detect_identity_milestones(self, identity: ProfessionalIdentity, 
                                   state: EnhancedAgentState) -> List[IdentityMilestone]:
        """Detect significant identity development milestones."""
        milestones = []
        
        # Role mastery milestone
        if (identity.role_confidence >= 0.8 and 
            not any(m.milestone_type == "role_mastery" for m in identity.milestone_history)):
            
            milestone = IdentityMilestone(
                milestone_id=str(uuid.uuid4()),
                milestone_type="role_mastery",
                description=f"Achieved mastery in {identity.primary_role} role",
                achieved_at=datetime.now(),
                role_context=identity.primary_role,
                impact_score=0.9,
                evidence_actions=self._get_recent_role_actions(state),
                metadata={"confidence_level": identity.role_confidence}
            )
            
            milestones.append(milestone)
            identity.milestone_history.append(milestone)
            self.logger.info(f"Role mastery milestone achieved for {identity.agent_id}")
        
        # Identity crisis milestone (large confidence drop)
        recent_confidence_changes = [point[1] for point in identity.growth_trajectory[-5:]]
        if (len(recent_confidence_changes) >= 3 and
            max(recent_confidence_changes) - min(recent_confidence_changes) > 0.3):
            
            milestone = IdentityMilestone(
                milestone_id=str(uuid.uuid4()),
                milestone_type="identity_crisis",
                description="Experienced significant identity uncertainty",
                achieved_at=datetime.now(),
                role_context=identity.primary_role,
                impact_score=0.7,
                evidence_actions=["confidence fluctuation detected"],
                metadata={"confidence_variance": max(recent_confidence_changes) - min(recent_confidence_changes)}
            )
            
            milestones.append(milestone)
            identity.milestone_history.append(milestone)
        
        # Role commitment milestone
        if (identity.role_commitment >= 0.8 and 
            not any(m.milestone_type == "role_commitment" for m in identity.milestone_history)):
            
            milestone = IdentityMilestone(
                milestone_id=str(uuid.uuid4()),
                milestone_type="role_commitment",
                description=f"Deep commitment to {identity.primary_role} role established",
                achieved_at=datetime.now(),
                role_context=identity.primary_role,
                impact_score=0.8,
                evidence_actions=self._get_recent_role_actions(state),
                metadata={"commitment_level": identity.role_commitment}
            )
            
            milestones.append(milestone)
            identity.milestone_history.append(milestone)
        
        return milestones
    
    def _get_recent_role_actions(self, state: EnhancedAgentState) -> List[str]:
        """Get recent actions that demonstrate role behavior."""
        # This would integrate with memory systems to get role-relevant actions
        # For now, return placeholder
        return ["demonstrated leadership", "showed expertise", "acted consistently with role"]
    
    def _track_identity_evolution(self, identity: ProfessionalIdentity, 
                                 state: EnhancedAgentState) -> bool:
        """Track significant identity evolution events."""
        evolution_detected = False
        
        # Check for role transitions in recent history
        recent_events = [event for event in identity.identity_history 
                        if event.get("event") == "role_change" and 
                        datetime.fromisoformat(event["timestamp"]) > 
                        datetime.now() - timedelta(days=1)]
        
        if recent_events:
            latest_change = recent_events[-1]
            
            evolution_record = IdentityEvolutionRecord(
                record_id=str(uuid.uuid4()),
                agent_id=identity.agent_id,
                from_role=latest_change["from_role"],
                to_role=latest_change["to_role"],
                transition_period=timedelta(days=1),  # Simplified
                catalyst_events=["role effectiveness change", "social feedback"],
                confidence_change=latest_change["confidence"] - 0.5,  # Simplified
                skill_transfers={},  # Would analyze skill transfer
                challenges_faced=["role adjustment", "identity integration"],
                support_received=["community acceptance"],
                recorded_at=datetime.now(),
                metadata={"transition_quality": "smooth"}
            )
            
            identity.evolution_records.append(evolution_record)
            evolution_detected = True
        
        return evolution_detected
    
    def _calculate_identity_coherence(self, identity: ProfessionalIdentity, 
                                     state: EnhancedAgentState) -> Dict[str, float]:
        """Calculate identity coherence metrics."""
        
        # Internal coherence: consistency between identity components
        internal_components = [
            identity.role_confidence,
            identity.role_commitment,
            identity.role_satisfaction
        ]
        internal_coherence = 1.0 - (statistics.pstdev(internal_components) if len(internal_components) > 1 else 0.0)
        
        # Behavioral consistency: would analyze action patterns vs role
        behavior_consistency = 0.7  # Placeholder - would analyze actual behavior
        
        # Value alignment: how well role aligns with agent's values
        value_alignment = 0.6  # Placeholder - would analyze value-role fit
        
        # Social validation: external feedback on identity performance
        social_validation = 0.5  # Placeholder - would analyze social responses
        
        # Update identity with calculated coherence
        identity.internal_coherence = internal_coherence
        identity.behavior_consistency = behavior_consistency
        identity.value_alignment = value_alignment
        identity.social_validation = social_validation
        
        return {
            "internal_coherence": internal_coherence,
            "behavior_consistency": behavior_consistency,
            "value_alignment": value_alignment,
            "social_validation": social_validation,
            "overall_coherence": (internal_coherence + behavior_consistency + 
                                value_alignment + social_validation) / 4
        }
    
    def _update_cross_session_continuity(self, identity: ProfessionalIdentity, 
                                        state: EnhancedAgentState) -> Dict[str, float]:
        """Update cross-session continuity metrics."""
        
        # Session consistency: how consistent identity is across sessions
        # This would compare current identity with stored identity
        session_consistency = 0.9  # Placeholder
        
        # Context adaptation: ability to maintain identity in different contexts
        context_adaptation = 0.7  # Placeholder
        
        # Memory integration: how well identity integrates with memory systems
        memory_integration = 0.8  # Placeholder
        
        identity.session_consistency_score = session_consistency
        identity.context_adaptation_ability = context_adaptation
        identity.memory_integration_quality = memory_integration
        
        return {
            "session_consistency": session_consistency,
            "context_adaptation": context_adaptation,
            "memory_integration": memory_integration
        }
    
    def _persist_identity(self, identity: ProfessionalIdentity) -> bool:
        """Persist identity to Store API and PostgreSQL."""
        try:
            # Cache locally
            self._cache_identity(identity)
            
            # Store in LangGraph Store API
            if self.store_integration and self.store_integration.store:
                identity_key = f"professional_identity_{identity.agent_id}"
                identity_data = {
                    **asdict(identity),
                    "created_at": identity.created_at.isoformat(),
                    "last_updated": identity.last_updated.isoformat(),
                    "formation_date": identity.formation_date.isoformat(),
                    "last_evolution_date": identity.last_evolution_date.isoformat() if identity.last_evolution_date else None,
                    "development_phase": identity.development_phase.value,
                    "identity_strength": identity.identity_strength.value,
                    "growth_trajectory": [(dt.isoformat(), score) for dt, score in identity.growth_trajectory]
                }
                
                # In real implementation, this would be:
                # await self.store_integration.store.aput("professional_identities", identity_key, identity_data)
            
            # Store in PostgreSQL for persistence
            if self.store_integration and self.store_integration.postgres_persistence:
                # In real implementation, would store in database
                pass
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to persist identity for {identity.agent_id}: {str(e)}")
            return False
    
    def _cache_identity(self, identity: ProfessionalIdentity) -> None:
        """Cache identity locally."""
        self.identity_cache[identity.agent_id] = identity
        cache_key = f"identity_{identity.agent_id}"
        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl)
    
    def _update_performance_stats(self, identity: ProfessionalIdentity, 
                                 milestones: List[IdentityMilestone],
                                 evolution_changes: bool) -> None:
        """Update module performance statistics."""
        self.persistence_stats["identity_updates"] += 1
        self.persistence_stats["milestones_recorded"] += len(milestones)
        
        if evolution_changes:
            self.persistence_stats["evolution_records"] += 1
        
        # Update average coherence
        current_coherence = identity.internal_coherence
        current_avg = self.persistence_stats["avg_identity_coherence"]
        update_count = self.persistence_stats["identity_updates"]
        
        self.persistence_stats["avg_identity_coherence"] = (
            (current_avg * (update_count - 1) + current_coherence) / update_count
        )
        
        self.persistence_stats["last_coherence_calculation"] = datetime.now().isoformat()
    
    def _calculate_development_progress(self, identity: ProfessionalIdentity) -> float:
        """Calculate overall development progress."""
        phase_progress = {
            IdentityDevelopmentPhase.EXPLORATION: 0.2,
            IdentityDevelopmentPhase.COMMITMENT: 0.4,
            IdentityDevelopmentPhase.SYNTHESIS: 0.6,
            IdentityDevelopmentPhase.MASTERY: 0.8,
            IdentityDevelopmentPhase.RENEWAL: 1.0
        }
        
        base_progress = phase_progress.get(identity.development_phase, 0.0)
        confidence_bonus = identity.role_confidence * 0.2
        
        return min(1.0, base_progress + confidence_bonus)
    
    def get_identity_summary(self) -> Dict[str, Any]:
        """
        Get summary of identity persistence module performance.
        
        Returns:
            Identity persistence summary with metrics
        """
        cached_identities = len(self.identity_cache)
        
        return {
            "module_name": self.module_name,
            "performance_stats": self.persistence_stats.copy(),
            "cache_status": {
                "cached_identities": cached_identities,
                "cache_hit_rate": "N/A",  # Would track this in real implementation
                "average_cache_age_minutes": "N/A"
            },
            "identity_development": {
                "avg_coherence": self.persistence_stats["avg_identity_coherence"],
                "milestone_rate": self.persistence_stats["milestones_recorded"] / max(self.persistence_stats["identity_updates"], 1),
                "evolution_rate": self.persistence_stats["evolution_records"] / max(self.persistence_stats["identity_updates"], 1)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of identity persistence module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8}
    )
    
    # Create identity persistence module
    identity_module = IdentityPersistenceModule(state_manager)
    
    print("Testing identity persistence...")
    
    # Process state to manage identity
    result = identity_module(state_manager.state)
    
    print(f"Identity result: {result}")
    
    # Get identity summary
    summary = identity_module.get_identity_summary()
    print(f"\nIdentity persistence summary:")
    print(f"- Performance stats: {summary['performance_stats']}")
    print(f"- Cache status: {summary['cache_status']}")
    print(f"- Development metrics: {summary['identity_development']}")
    
    print("Identity persistence module example completed!")