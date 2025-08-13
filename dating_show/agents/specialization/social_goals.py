#!/usr/bin/env python3
"""
Social Goal Interpretation System - Enhanced PIANO Architecture
Interprets community roles and collective goal contributions for advanced social dynamics

Implementation for Task 3.1.3: Social Goal Interpretation  
- Community role recognition
- Social expectation alignment
- Collective goal contribution measurement
- Real-time performance <50ms with social context awareness
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SocialGoalResult:
    """Result of social goal interpretation analysis"""
    agent_id: str
    community_role: str  # Primary social role in community
    social_contribution_score: float  # 0.0 to 1.0
    collective_goal_alignment: Dict[str, float]  # Goal type -> alignment score
    social_expectations: List[Dict[str, Any]]  # Expected behaviors for role
    community_impact_metrics: Dict[str, float]  # Various impact measurements
    social_network_position: Dict[str, Any]  # Position in social network
    collaboration_patterns: List[Dict[str, Any]]  # Collaboration analysis
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def is_socially_aligned(self, threshold: float = 0.7) -> bool:
        """Check if socially aligned above threshold"""
        return self.social_contribution_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), indent=2)


class CommunityRoleRecognizer:
    """Recognizes social roles within community context"""
    
    def __init__(self):
        self.community_roles = {
            "leader": {
                "keywords": ["lead", "guide", "direct", "manage", "coordinate", "organize", "inspire"],
                "actions": ["coordinate_team", "lead_meeting", "delegate_task", "make_decision", "set_direction"],
                "social_traits": ["influential", "decisive", "responsible", "visionary"],
                "network_position": "central",
                "interaction_patterns": ["one_to_many", "directive", "supportive"],
                "weight": 1.0
            },
            "mediator": {
                "keywords": ["mediate", "resolve", "negotiate", "facilitate", "harmonize", "bridge"],
                "actions": ["mediate_conflict", "facilitate_discussion", "build_consensus", "negotiate", "resolve_dispute"],
                "social_traits": ["diplomatic", "empathetic", "balanced", "trustworthy"],
                "network_position": "bridge",
                "interaction_patterns": ["many_to_many", "facilitative", "neutral"],
                "weight": 1.0
            },
            "connector": {
                "keywords": ["connect", "introduce", "network", "link", "bridge", "collaborate"],
                "actions": ["introduce_people", "organize_social_event", "facilitate_connections", "network", "collaborate"],
                "social_traits": ["social", "outgoing", "friendly", "inclusive"],
                "network_position": "hub",
                "interaction_patterns": ["many_to_many", "inclusive", "energetic"],
                "weight": 1.0
            },
            "supporter": {
                "keywords": ["support", "help", "assist", "encourage", "enable", "backup"],
                "actions": ["provide_support", "assist_others", "encourage", "help_with_task", "offer_resources"],
                "social_traits": ["helpful", "reliable", "caring", "loyal"],
                "network_position": "peripheral_active",
                "interaction_patterns": ["one_to_one", "supportive", "responsive"],
                "weight": 1.0
            },
            "innovator": {
                "keywords": ["innovate", "create", "develop", "experiment", "pioneer", "discover"],
                "actions": ["propose_idea", "experiment", "create_solution", "innovate", "research"],
                "social_traits": ["creative", "curious", "independent", "forward_thinking"],
                "network_position": "specialized",
                "interaction_patterns": ["selective", "idea_sharing", "consultative"],
                "weight": 1.0
            },
            "observer": {
                "keywords": ["observe", "monitor", "watch", "analyze", "study", "assess"],
                "actions": ["observe_interactions", "monitor_situation", "analyze_behavior", "provide_feedback", "assess_group"],
                "social_traits": ["analytical", "thoughtful", "perceptive", "quiet"],
                "network_position": "peripheral",
                "interaction_patterns": ["observational", "analytical", "selective"],
                "weight": 0.8
            }
        }
    
    def analyze_social_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in social interactions"""
        if not interactions:
            return {"interaction_frequency": 0, "interaction_types": {}, "network_metrics": {}}
        
        interaction_types = Counter()
        interaction_partners = set()
        interaction_directions = {"initiated": 0, "responded": 0}
        
        for interaction in interactions:
            interaction_type = interaction.get("type", "unknown")
            interaction_types[interaction_type] += 1
            
            # Track interaction partners
            partners = interaction.get("participants", [])
            interaction_partners.update(partners)
            
            # Track interaction direction
            if interaction.get("initiator") == interaction.get("agent_id"):
                interaction_directions["initiated"] += 1
            else:
                interaction_directions["responded"] += 1
        
        # Calculate network metrics
        network_metrics = {
            "connectivity": len(interaction_partners),
            "interaction_diversity": len(interaction_types),
            "initiative_ratio": interaction_directions["initiated"] / len(interactions) if interactions else 0,
            "response_ratio": interaction_directions["responded"] / len(interactions) if interactions else 0
        }
        
        return {
            "interaction_frequency": len(interactions),
            "interaction_types": dict(interaction_types),
            "network_metrics": network_metrics,
            "unique_partners": len(interaction_partners)
        }
    
    def recognize_community_role(self, agent_data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Recognize primary community role from agent behavior and interactions"""
        action_history = agent_data.get("action_history", [])
        interactions = agent_data.get("social_interactions", [])
        communication_style = agent_data.get("communication_style", {})
        
        role_scores = {}
        supporting_evidence = defaultdict(list)
        
        # Analyze actions against role patterns
        action_types = [action["action"] for action in action_history]
        
        for role_name, role_config in self.community_roles.items():
            score = 0.0
            
            # Score based on action alignment
            for action in action_types:
                if action in role_config["actions"]:
                    score += role_config["weight"]
                
                # Check for keyword matches in action names
                for keyword in role_config["keywords"]:
                    if keyword in action.lower():
                        score += role_config["weight"] * 0.5
                        supporting_evidence[role_name].append(f"action_keyword_{keyword}")
            
            # Score based on interaction patterns
            interaction_analysis = self.analyze_social_interactions(interactions)
            
            if role_name == "leader" and interaction_analysis["network_metrics"].get("initiative_ratio", 0) > 0.6:
                score += 2.0
                supporting_evidence[role_name].append("high_initiative_interactions")
            
            if role_name == "connector" and interaction_analysis.get("unique_partners", 0) > 5:
                score += 2.0
                supporting_evidence[role_name].append("high_connectivity")
            
            if role_name == "mediator" and "mediate" in str(action_types).lower():
                score += 2.0
                supporting_evidence[role_name].append("mediation_actions")
            
            if role_name == "supporter" and interaction_analysis["network_metrics"].get("response_ratio", 0) > 0.7:
                score += 1.5
                supporting_evidence[role_name].append("high_responsiveness")
            
            if role_name == "observer" and interaction_analysis.get("interaction_frequency", 0) < 3:
                score += 1.0
                supporting_evidence[role_name].append("low_interaction_frequency")
            
            # Normalize score
            if action_types:
                score = score / len(action_types)
            
            role_scores[role_name] = score
        
        # Determine primary role
        if not role_scores or max(role_scores.values()) == 0:
            return "undefined", 0.0, {}
        
        primary_role = max(role_scores.items(), key=lambda x: x[1])
        role_name, confidence = primary_role
        
        return role_name, confidence, dict(supporting_evidence)


class SocialExpectationAnalyzer:
    """Analyzes alignment with social expectations for roles"""
    
    def __init__(self):
        self.role_expectations = {
            "leader": {
                "expected_behaviors": ["make_decisions", "provide_direction", "take_responsibility", "inspire_others"],
                "communication_style": ["clear", "confident", "directive", "supportive"],
                "interaction_frequency": "high",
                "network_position": "central"
            },
            "mediator": {
                "expected_behaviors": ["listen_actively", "remain_neutral", "find_compromise", "build_bridges"],
                "communication_style": ["diplomatic", "balanced", "empathetic", "patient"],
                "interaction_frequency": "medium",
                "network_position": "bridge"
            },
            "connector": {
                "expected_behaviors": ["introduce_people", "facilitate_connections", "organize_events", "share_information"],
                "communication_style": ["friendly", "enthusiastic", "inclusive", "energetic"],
                "interaction_frequency": "high",
                "network_position": "hub"
            },
            "supporter": {
                "expected_behaviors": ["offer_help", "provide_resources", "encourage_others", "be_reliable"],
                "communication_style": ["caring", "responsive", "helpful", "loyal"],
                "interaction_frequency": "medium",
                "network_position": "active"
            },
            "innovator": {
                "expected_behaviors": ["propose_ideas", "experiment", "think_creatively", "challenge_status_quo"],
                "communication_style": ["creative", "questioning", "independent", "visionary"],
                "interaction_frequency": "selective",
                "network_position": "specialized"
            },
            "observer": {
                "expected_behaviors": ["monitor_situations", "provide_insights", "analyze_patterns", "give_feedback"],
                "communication_style": ["thoughtful", "analytical", "precise", "quiet"],
                "interaction_frequency": "low",
                "network_position": "peripheral"
            }
        }
    
    def analyze_expectation_alignment(self, agent_data: Dict[str, Any], community_role: str) -> Dict[str, float]:
        """Analyze how well agent meets social expectations for their role"""
        if community_role not in self.role_expectations:
            return {"behavior_alignment": 0.0, "communication_alignment": 0.0, "overall_alignment": 0.0}
        
        expectations = self.role_expectations[community_role]
        action_history = agent_data.get("action_history", [])
        communication_style = agent_data.get("communication_style", {})
        interactions = agent_data.get("social_interactions", [])
        
        # Analyze behavior alignment
        behavior_alignment = self._analyze_behavior_alignment(action_history, expectations["expected_behaviors"])
        
        # Analyze communication alignment
        communication_alignment = self._analyze_communication_alignment(communication_style, expectations["communication_style"])
        
        # Analyze interaction pattern alignment
        interaction_alignment = self._analyze_interaction_alignment(interactions, expectations)
        
        overall_alignment = np.mean([behavior_alignment, communication_alignment, interaction_alignment])
        
        return {
            "behavior_alignment": behavior_alignment,
            "communication_alignment": communication_alignment,
            "interaction_alignment": interaction_alignment,
            "overall_alignment": overall_alignment
        }
    
    def _analyze_behavior_alignment(self, actions: List[Dict[str, Any]], expected_behaviors: List[str]) -> float:
        """Analyze alignment between actions and expected behaviors"""
        if not actions or not expected_behaviors:
            return 0.0
        
        action_types = [action["action"] for action in actions]
        aligned_behaviors = 0
        
        for behavior in expected_behaviors:
            # Check for direct matches or keyword matches
            for action in action_types:
                if behavior.replace("_", " ") in action.replace("_", " ").lower():
                    aligned_behaviors += 1
                    break
                elif any(word in action.lower() for word in behavior.split("_")):
                    aligned_behaviors += 0.5
                    break
        
        return aligned_behaviors / len(expected_behaviors)
    
    def _analyze_communication_alignment(self, style: Dict[str, Any], expected_style: List[str]) -> float:
        """Analyze alignment between communication style and expectations"""
        if not style or not expected_style:
            return 0.5  # Neutral score if no data
        
        style_matches = 0
        for expected_trait in expected_style:
            if expected_trait in style:
                trait_score = style.get(expected_trait, 0.0)
                if isinstance(trait_score, (int, float)) and trait_score > 0.5:
                    style_matches += 1
                elif trait_score is True:
                    style_matches += 1
        
        return style_matches / len(expected_style) if expected_style else 0.0
    
    def _analyze_interaction_alignment(self, interactions: List[Dict[str, Any]], expectations: Dict[str, Any]) -> float:
        """Analyze alignment between interaction patterns and expectations"""
        if not interactions:
            return 0.5  # Neutral score if no interaction data
        
        expected_frequency = expectations.get("interaction_frequency", "medium")
        interaction_count = len(interactions)
        
        # Define frequency thresholds
        frequency_thresholds = {
            "low": (0, 3),
            "medium": (3, 8),
            "high": (8, float('inf'))
        }
        
        min_threshold, max_threshold = frequency_thresholds.get(expected_frequency, (0, float('inf')))
        
        if min_threshold <= interaction_count <= max_threshold:
            return 1.0
        elif interaction_count < min_threshold:
            return max(0.0, interaction_count / min_threshold)
        else:  # interaction_count > max_threshold
            return max(0.0, max_threshold / interaction_count)


class CollectiveGoalAnalyzer:
    """Analyzes contribution to collective community goals"""
    
    def __init__(self):
        self.collective_goal_types = {
            "community_harmony": {
                "keywords": ["harmony", "peace", "cooperation", "unity", "consensus"],
                "contributing_actions": ["mediate_conflict", "build_consensus", "facilitate_discussion", "resolve_dispute"],
                "weight": 1.0
            },
            "community_growth": {
                "keywords": ["growth", "development", "expansion", "improvement", "progress"],
                "contributing_actions": ["organize_event", "recruit_members", "develop_resources", "improve_processes"],
                "weight": 1.0
            },
            "knowledge_sharing": {
                "keywords": ["share", "teach", "learn", "educate", "knowledge", "information"],
                "contributing_actions": ["share_information", "teach_skill", "create_documentation", "mentor"],
                "weight": 1.0
            },
            "innovation": {
                "keywords": ["innovate", "create", "develop", "experiment", "new", "creative"],
                "contributing_actions": ["propose_idea", "experiment", "create_solution", "research"],
                "weight": 1.0
            },
            "resource_optimization": {
                "keywords": ["optimize", "efficient", "resource", "manage", "allocate", "improve"],
                "contributing_actions": ["optimize_process", "manage_resources", "improve_efficiency", "allocate_resources"],
                "weight": 1.0
            }
        }
    
    def measure_collective_contribution(self, agent_data: Dict[str, Any], 
                                      community_goals: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure agent's contribution to collective community goals"""
        action_history = agent_data.get("action_history", [])
        
        if community_goals is None:
            # Use default collective goals if none provided
            community_goals = [{"goal_type": goal_type} for goal_type in self.collective_goal_types.keys()]
        
        contributions = {}
        
        for goal in community_goals:
            goal_type = goal.get("goal_type", "")
            if goal_type not in self.collective_goal_types:
                continue
            
            goal_config = self.collective_goal_types[goal_type]
            contribution_score = 0.0
            
            # Check actions that contribute to this goal
            for action in action_history:
                action_type = action["action"]
                
                if action_type in goal_config["contributing_actions"]:
                    contribution_score += goal_config["weight"]
                
                # Check for keyword matches
                for keyword in goal_config["keywords"]:
                    if keyword in action_type.lower():
                        contribution_score += goal_config["weight"] * 0.5
            
            # Normalize by total actions
            if action_history:
                contribution_score = contribution_score / len(action_history)
            
            contributions[goal_type] = contribution_score
        
        return contributions
    
    def analyze_collaboration_patterns(self, agent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns of collaboration and teamwork"""
        interactions = agent_data.get("social_interactions", [])
        action_history = agent_data.get("action_history", [])
        
        collaboration_patterns = []
        
        # Analyze group interactions
        group_interactions = [i for i in interactions if len(i.get("participants", [])) > 2]
        
        if group_interactions:
            collaboration_patterns.append({
                "pattern_type": "group_collaboration",
                "frequency": len(group_interactions),
                "strength": len(group_interactions) / len(interactions) if interactions else 0,
                "description": "Participates in group activities and discussions"
            })
        
        # Analyze coordination actions
        coordination_actions = [a for a in action_history if "coordinate" in a["action"].lower()]
        
        if coordination_actions:
            collaboration_patterns.append({
                "pattern_type": "coordination",
                "frequency": len(coordination_actions),
                "strength": len(coordination_actions) / len(action_history) if action_history else 0,
                "description": "Takes initiative in coordinating group activities"
            })
        
        # Analyze supportive actions
        supportive_actions = [a for a in action_history if any(
            keyword in a["action"].lower() 
            for keyword in ["help", "support", "assist", "collaborate"]
        )]
        
        if supportive_actions:
            collaboration_patterns.append({
                "pattern_type": "supportive_collaboration",
                "frequency": len(supportive_actions),
                "strength": len(supportive_actions) / len(action_history) if action_history else 0,
                "description": "Provides support and assistance to community members"
            })
        
        return sorted(collaboration_patterns, key=lambda x: x["strength"], reverse=True)


class SocialGoalInterpreter:
    """Main class for social goal interpretation"""
    
    def __init__(self, community_context: Dict[str, Any] = None):
        self.community_context = community_context or {}
        
        self.role_recognizer = CommunityRoleRecognizer()
        self.expectation_analyzer = SocialExpectationAnalyzer()
        self.collective_analyzer = CollectiveGoalAnalyzer()
        
        logger.info("SocialGoalInterpreter initialized")
    
    def interpret_social_goals(self, agent_data: Dict[str, Any]) -> SocialGoalResult:
        """
        Interpret agent's social goals and community role
        
        Args:
            agent_data: Dictionary containing agent behavior, interactions, and context
            
        Returns:
            SocialGoalResult with comprehensive social analysis
        """
        start_time = time.time()
        
        agent_id = agent_data.get("agent_id", "unknown")
        
        try:
            # Recognize community role
            community_role, role_confidence, role_evidence = self.role_recognizer.recognize_community_role(agent_data)
            
            # Analyze social expectations alignment
            expectation_alignment = self.expectation_analyzer.analyze_expectation_alignment(agent_data, community_role)
            
            # Measure collective goal contributions
            collective_contributions = self.collective_analyzer.measure_collective_contribution(
                agent_data, 
                self.community_context.get("collective_goals")
            )
            
            # Analyze collaboration patterns
            collaboration_patterns = self.collective_analyzer.analyze_collaboration_patterns(agent_data)
            
            # Calculate overall social contribution score
            social_contribution_score = np.mean([
                role_confidence,
                expectation_alignment.get("overall_alignment", 0.0),
                np.mean(list(collective_contributions.values())) if collective_contributions else 0.0
            ])
            
            # Analyze social network position
            interactions = agent_data.get("social_interactions", [])
            network_analysis = self.role_recognizer.analyze_social_interactions(interactions)
            
            social_network_position = {
                "centrality": network_analysis["network_metrics"].get("connectivity", 0),
                "activity_level": network_analysis.get("interaction_frequency", 0),
                "diversity": network_analysis["network_metrics"].get("interaction_diversity", 0),
                "role_based_position": self.role_recognizer.community_roles.get(community_role, {}).get("network_position", "undefined")
            }
            
            # Generate social expectations for the role
            role_config = self.role_recognizer.community_roles.get(community_role, {})
            social_expectations = [
                {
                    "expectation_type": "behavior",
                    "expected_behaviors": role_config.get("actions", []),
                    "alignment_score": expectation_alignment.get("behavior_alignment", 0.0)
                },
                {
                    "expectation_type": "communication",
                    "expected_style": role_config.get("social_traits", []),
                    "alignment_score": expectation_alignment.get("communication_alignment", 0.0)
                }
            ]
            
            # Calculate community impact metrics
            community_impact_metrics = {
                "role_fulfillment": min(1.0, role_confidence),
                "expectation_alignment": min(1.0, expectation_alignment.get("overall_alignment", 0.0)),
                "collective_contribution": min(1.0, np.mean(list(collective_contributions.values())) if collective_contributions else 0.0),
                "collaboration_effectiveness": min(1.0, np.mean([p["strength"] for p in collaboration_patterns]) if collaboration_patterns else 0.0),
                "social_influence": min(1.0, social_network_position["centrality"] / 10.0)  # Normalized influence
            }
            
            # Check performance
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 50:
                logger.warning(f"Social goal interpretation for {agent_id} took {processing_time:.2f}ms > 50ms")
            
            result = SocialGoalResult(
                agent_id=agent_id,
                community_role=community_role,
                social_contribution_score=social_contribution_score,
                collective_goal_alignment=collective_contributions,
                social_expectations=social_expectations,
                community_impact_metrics=community_impact_metrics,
                social_network_position=social_network_position,
                collaboration_patterns=collaboration_patterns
            )
            
            logger.info(f"Social goal interpretation for {agent_id}: role={community_role}, "
                       f"contribution={social_contribution_score:.3f} (time={processing_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in social goal interpretation for {agent_id}: {str(e)}")
            return SocialGoalResult(
                agent_id=agent_id,
                community_role="undefined",
                social_contribution_score=0.0,
                collective_goal_alignment={},
                social_expectations=[],
                community_impact_metrics={},
                social_network_position={},
                collaboration_patterns=[]
            )
    
    def batch_interpret_social_goals(self, agents_data: List[Dict[str, Any]]) -> List[SocialGoalResult]:
        """Interpret social goals for multiple agents"""
        results = []
        start_time = time.time()
        
        for agent_data in agents_data:
            result = self.interpret_social_goals(agent_data)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(agents_data) if agents_data else 0
        
        logger.info(f"Batch social goal interpretation completed: {len(agents_data)} agents, "
                   f"avg {avg_time:.2f}ms per agent")
        
        return results
    
    def get_social_statistics(self, results: List[SocialGoalResult]) -> Dict[str, Any]:
        """Calculate statistics from social goal interpretation results"""
        if not results:
            return {}
        
        role_distribution = Counter([r.community_role for r in results])
        socially_aligned = [r for r in results if r.is_socially_aligned()]
        
        return {
            "total_agents": len(results),
            "socially_aligned_agents": len(socially_aligned),
            "social_alignment_rate": len(socially_aligned) / len(results),
            "role_distribution": dict(role_distribution),
            "avg_social_contribution": np.mean([r.social_contribution_score for r in results]),
            "community_impact_summary": {
                "avg_role_fulfillment": np.mean([r.community_impact_metrics.get("role_fulfillment", 0) for r in results]),
                "avg_collaboration": np.mean([r.community_impact_metrics.get("collaboration_effectiveness", 0) for r in results]),
                "avg_collective_contribution": np.mean([r.community_impact_metrics.get("collective_contribution", 0) for r in results])
            }
        }


# Export main classes
__all__ = [
    'SocialGoalInterpreter',
    'CommunityRoleRecognizer',
    'SocialExpectationAnalyzer',
    'CollectiveGoalAnalyzer',
    'SocialGoalResult'
]


if __name__ == "__main__":
    # Example usage and testing
    interpreter = SocialGoalInterpreter()
    
    # Test with sample agent data
    sample_agent = {
        "agent_id": "social_agent_001",
        "action_history": [
            {"action": "organize_social_event", "timestamp": time.time() - 3600, "success": True},
            {"action": "introduce_people", "timestamp": time.time() - 3000, "success": True},
            {"action": "facilitate_connections", "timestamp": time.time() - 2400, "success": True},
            {"action": "network", "timestamp": time.time() - 1800, "success": True},
            {"action": "organize_social_event", "timestamp": time.time() - 1200, "success": True},
            {"action": "collaborate", "timestamp": time.time() - 600, "success": True},
            {"action": "introduce_people", "timestamp": time.time(), "success": True},
        ],
        "social_interactions": [
            {
                "type": "group_discussion",
                "participants": ["agent_001", "agent_002", "agent_003", "agent_004"],
                "initiator": "social_agent_001",
                "timestamp": time.time() - 2000
            },
            {
                "type": "introduction",
                "participants": ["agent_002", "agent_005"],
                "initiator": "social_agent_001",
                "timestamp": time.time() - 1500
            }
        ],
        "communication_style": {
            "friendly": 0.9,
            "enthusiastic": 0.8,
            "inclusive": 0.9,
            "energetic": 0.7
        }
    }
    
    result = interpreter.interpret_social_goals(sample_agent)
    print("Social Goal Interpretation Result:")
    print(result.to_json())