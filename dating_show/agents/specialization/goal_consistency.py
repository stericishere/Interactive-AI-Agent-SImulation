#!/usr/bin/env python3
"""
Goal Consistency Measurement System - Enhanced PIANO Architecture
Measures consistency between agent goals and actions to reinforce role detection

Implementation for Task 3.1.2: Goal Consistency Measurement
- Goal tracking and consistency scoring
- Role-goal alignment validation
- Consistency-based role reinforcement
- Real-time performance <50ms with >80% accuracy
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GoalConsistencyResult:
    """Result of goal consistency measurement"""
    agent_id: str
    consistency_score: float  # 0.0 to 1.0
    goal_alignment_scores: Dict[str, float]  # Goal type -> alignment score
    inconsistent_actions: List[Dict[str, Any]]  # Actions that don't align with goals
    dominant_goal_patterns: List[Dict[str, Any]]  # Most consistent goal patterns
    temporal_consistency: float  # Consistency over time
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def is_consistent(self, threshold: float = 0.7) -> bool:
        """Check if goals are consistent above threshold"""
        return self.consistency_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), indent=2)


class GoalTracker:
    """Tracks agent goals and their evolution over time"""
    
    def __init__(self, max_goal_history: int = 50):
        self.max_goal_history = max_goal_history
        self.goal_categories = {
            "social": {
                "keywords": ["socialize", "relationship", "friend", "connect", "network", "collaborate"],
                "actions": ["socialize", "build_relationship", "network", "collaborate", "organize_social_event"],
                "weight": 1.0
            },
            "achievement": {
                "keywords": ["achieve", "accomplish", "succeed", "complete", "finish", "goal", "target"],
                "actions": ["complete_task", "achieve_goal", "finish_project", "succeed_in"],
                "weight": 1.0
            },
            "management": {
                "keywords": ["manage", "lead", "coordinate", "organize", "plan", "control", "supervise"],
                "actions": ["plan_event", "coordinate_team", "delegate_task", "manage_resources", "evaluate_performance"],
                "weight": 1.0
            },
            "analytical": {
                "keywords": ["analyze", "research", "investigate", "study", "examine", "evaluate", "assess"],
                "actions": ["analyze_data", "research", "investigate", "evaluate_options", "create_report"],
                "weight": 1.0
            },
            "efficiency": {
                "keywords": ["optimize", "efficient", "improve", "enhance", "streamline", "effective"],
                "actions": ["optimize_process", "improve_efficiency", "streamline_workflow", "enhance_performance"],
                "weight": 1.0
            }
        }
    
    def extract_goals_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract goal information from text descriptions"""
        if not text:
            return []
        
        text_lower = text.lower()
        goals = []
        
        for goal_type, config in self.goal_categories.items():
            relevance_score = 0.0
            matched_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    relevance_score += config["weight"]
                    matched_keywords.append(keyword)
            
            if relevance_score > 0:
                goals.append({
                    "goal_type": goal_type,
                    "relevance_score": relevance_score,
                    "matched_keywords": matched_keywords,
                    "extracted_from": "text_analysis",
                    "timestamp": time.time()
                })
        
        return sorted(goals, key=lambda x: x["relevance_score"], reverse=True)
    
    def infer_goals_from_actions(self, action_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer goals from action patterns"""
        if not action_history:
            return []
        
        action_counts = defaultdict(int)
        for action in action_history:
            action_counts[action["action"]] += 1
        
        inferred_goals = []
        
        for goal_type, config in self.goal_categories.items():
            goal_relevance = 0.0
            supporting_actions = []
            
            for action_type, count in action_counts.items():
                if action_type in config["actions"]:
                    goal_relevance += count * config["weight"]
                    supporting_actions.append({
                        "action": action_type,
                        "frequency": count
                    })
            
            if goal_relevance > 0:
                inferred_goals.append({
                    "goal_type": goal_type,
                    "relevance_score": goal_relevance / len(action_history),  # Normalize by action count
                    "supporting_actions": supporting_actions,
                    "extracted_from": "action_analysis",
                    "timestamp": time.time()
                })
        
        return sorted(inferred_goals, key=lambda x: x["relevance_score"], reverse=True)
    
    def track_goal_evolution(self, goal_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track how goals evolve over time"""
        if not goal_history:
            return {"evolution_patterns": [], "stability_score": 0.0}
        
        # Sort by timestamp
        sorted_goals = sorted(goal_history, key=lambda x: x.get("timestamp", 0))
        
        # Analyze goal type changes over time
        goal_sequence = [goal["goal_type"] for goal in sorted_goals]
        
        # Calculate stability (how often goals change)
        changes = sum(1 for i in range(1, len(goal_sequence)) if goal_sequence[i] != goal_sequence[i-1])
        stability_score = 1.0 - (changes / max(len(goal_sequence) - 1, 1))
        
        # Identify evolution patterns
        evolution_patterns = []
        current_pattern = []
        
        for i, goal_type in enumerate(goal_sequence):
            if not current_pattern or goal_type == current_pattern[-1]["goal_type"]:
                if current_pattern and goal_type == current_pattern[-1]["goal_type"]:
                    current_pattern[-1]["duration"] += 1
                else:
                    current_pattern.append({
                        "goal_type": goal_type,
                        "start_index": i,
                        "duration": 1
                    })
            else:
                if len(current_pattern) > 0:
                    evolution_patterns.append(current_pattern)
                current_pattern = [{
                    "goal_type": goal_type,
                    "start_index": i,
                    "duration": 1
                }]
        
        if current_pattern:
            evolution_patterns.append(current_pattern)
        
        return {
            "evolution_patterns": evolution_patterns,
            "stability_score": stability_score,
            "goal_changes": changes,
            "dominant_goals": self._find_dominant_goals(goal_sequence)
        }
    
    def _find_dominant_goals(self, goal_sequence: List[str]) -> List[Dict[str, Any]]:
        """Find the most dominant goal types"""
        goal_counts = defaultdict(int)
        for goal_type in goal_sequence:
            goal_counts[goal_type] += 1
        
        total_goals = len(goal_sequence)
        dominant_goals = []
        
        for goal_type, count in goal_counts.items():
            dominance_score = count / total_goals
            dominant_goals.append({
                "goal_type": goal_type,
                "dominance_score": dominance_score,
                "occurrence_count": count
            })
        
        return sorted(dominant_goals, key=lambda x: x["dominance_score"], reverse=True)


class ConsistencyAnalyzer:
    """Analyzes consistency between goals and actions"""
    
    def __init__(self, consistency_window: int = 20):
        self.consistency_window = consistency_window
        self.goal_tracker = GoalTracker()
        
    def calculate_goal_action_alignment(self, goals: List[Dict[str, Any]], 
                                      actions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate alignment between stated goals and actual actions"""
        if not goals or not actions:
            return {}
        
        alignment_scores = {}
        action_types = [action["action"] for action in actions]
        
        for goal in goals:
            goal_type = goal["goal_type"]
            goal_config = self.goal_tracker.goal_categories.get(goal_type, {})
            expected_actions = goal_config.get("actions", [])
            
            if not expected_actions:
                alignment_scores[goal_type] = 0.0
                continue
            
            # Count how many actions align with this goal
            aligned_actions = sum(1 for action in action_types if action in expected_actions)
            alignment_score = aligned_actions / len(action_types)
            
            # Weight by goal relevance
            goal_relevance = goal.get("relevance_score", 1.0)
            weighted_alignment = alignment_score * min(goal_relevance, 1.0)
            
            alignment_scores[goal_type] = weighted_alignment
        
        return alignment_scores
    
    def identify_inconsistent_actions(self, goals: List[Dict[str, Any]], 
                                    actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify actions that don't align with stated goals"""
        if not goals or not actions:
            return []
        
        # Get all expected actions from goals
        expected_actions = set()
        for goal in goals:
            goal_type = goal["goal_type"]
            goal_config = self.goal_tracker.goal_categories.get(goal_type, {})
            expected_actions.update(goal_config.get("actions", []))
        
        # Find actions that don't match any goal
        inconsistent_actions = []
        for action in actions:
            if action["action"] not in expected_actions:
                inconsistent_actions.append({
                    **action,
                    "inconsistency_reason": "action_not_aligned_with_goals",
                    "expected_actions": list(expected_actions)
                })
        
        return inconsistent_actions
    
    def calculate_temporal_consistency(self, agent_history: List[Dict[str, Any]]) -> float:
        """Calculate consistency of goal-action alignment over time"""
        if not agent_history:
            return 0.0
        
        # Sort by timestamp
        sorted_history = sorted(agent_history, key=lambda x: x.get("timestamp", 0))
        
        # Calculate consistency in sliding windows
        consistency_scores = []
        window_size = min(self.consistency_window, len(sorted_history))
        
        for i in range(len(sorted_history) - window_size + 1):
            window_data = sorted_history[i:i + window_size]
            
            # Extract goals and actions from window
            window_goals = []
            window_actions = []
            
            for entry in window_data:
                if entry.get("type") == "goal":
                    window_goals.append(entry)
                elif entry.get("type") == "action":
                    window_actions.append(entry)
            
            if window_goals and window_actions:
                alignment_scores = self.calculate_goal_action_alignment(window_goals, window_actions)
                if alignment_scores:
                    avg_alignment = np.mean(list(alignment_scores.values()))
                    consistency_scores.append(avg_alignment)
        
        if not consistency_scores:
            return 0.0
        
        return np.mean(consistency_scores)
    
    def analyze_goal_pattern_strength(self, goals: List[Dict[str, Any]], 
                                    actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze strength of different goal patterns"""
        if not goals:
            return []
        
        pattern_strengths = []
        
        for goal in goals:
            goal_type = goal["goal_type"]
            goal_config = self.goal_tracker.goal_categories.get(goal_type, {})
            expected_actions = set(goal_config.get("actions", []))
            
            if not expected_actions:
                continue
            
            # Calculate pattern strength based on action alignment
            action_types = set(action["action"] for action in actions)
            matching_actions = expected_actions.intersection(action_types)
            
            strength = len(matching_actions) / len(expected_actions) if expected_actions else 0.0
            frequency = goal.get("relevance_score", 0.0)
            
            pattern_strengths.append({
                "goal_type": goal_type,
                "pattern_strength": strength,
                "goal_frequency": frequency,
                "combined_score": strength * frequency,
                "matching_actions": list(matching_actions),
                "expected_actions": list(expected_actions)
            })
        
        return sorted(pattern_strengths, key=lambda x: x["combined_score"], reverse=True)


class GoalConsistencyMeasurement:
    """Main class for measuring goal consistency"""
    
    def __init__(self, consistency_threshold: float = 0.7, 
                 temporal_window: int = 20):
        self.consistency_threshold = consistency_threshold
        self.temporal_window = temporal_window
        
        self.goal_tracker = GoalTracker()
        self.consistency_analyzer = ConsistencyAnalyzer(temporal_window)
        
        logger.info(f"GoalConsistencyMeasurement initialized with threshold={consistency_threshold}")
    
    def measure_consistency(self, agent_data: Dict[str, Any]) -> GoalConsistencyResult:
        """
        Measure goal consistency for an agent
        
        Args:
            agent_data: Dictionary containing agent_id, goals, action_history, etc.
            
        Returns:
            GoalConsistencyResult with consistency metrics
        """
        start_time = time.time()
        
        agent_id = agent_data.get("agent_id", "unknown")
        action_history = agent_data.get("action_history", [])
        stated_goals = agent_data.get("goals", [])
        goal_text = agent_data.get("goal_description", "")
        
        try:
            # Extract goals from various sources
            text_goals = self.goal_tracker.extract_goals_from_text(goal_text) if goal_text else []
            action_goals = self.goal_tracker.infer_goals_from_actions(action_history)
            
            # Combine all goals
            all_goals = stated_goals + text_goals + action_goals
            
            if not all_goals:
                logger.warning(f"No goals found for agent {agent_id}")
                return GoalConsistencyResult(
                    agent_id=agent_id,
                    consistency_score=0.0,
                    goal_alignment_scores={},
                    inconsistent_actions=[],
                    dominant_goal_patterns=[],
                    temporal_consistency=0.0
                )
            
            # Calculate goal-action alignment
            alignment_scores = self.consistency_analyzer.calculate_goal_action_alignment(all_goals, action_history)
            
            # Identify inconsistent actions
            inconsistent_actions = self.consistency_analyzer.identify_inconsistent_actions(all_goals, action_history)
            
            # Analyze goal pattern strengths
            pattern_strengths = self.consistency_analyzer.analyze_goal_pattern_strength(all_goals, action_history)
            
            # Calculate temporal consistency
            temporal_consistency = self.consistency_analyzer.calculate_temporal_consistency(
                agent_data.get("history", [])
            )
            
            # Calculate overall consistency score
            if alignment_scores:
                # Use weighted average based on goal relevance
                weighted_sum = 0.0
                total_weight = 0.0
                
                for goal in all_goals:
                    goal_type = goal["goal_type"]
                    if goal_type in alignment_scores:
                        relevance = goal.get("relevance_score", 1.0)
                        weighted_sum += alignment_scores[goal_type] * relevance
                        total_weight += relevance
                
                consistency_score = weighted_sum / total_weight if total_weight > 0 else np.mean(list(alignment_scores.values()))
                
                # Less harsh penalty for inconsistent actions
                if action_history:
                    inconsistency_penalty = len(inconsistent_actions) / len(action_history)
                    consistency_score = max(0.0, consistency_score - inconsistency_penalty * 0.1)
                
                # Boost for temporal consistency
                consistency_score = min(1.0, consistency_score + temporal_consistency * 0.3)
            else:
                consistency_score = 0.0
            
            # Check performance
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 50:
                logger.warning(f"Goal consistency measurement for {agent_id} took {processing_time:.2f}ms > 50ms")
            
            result = GoalConsistencyResult(
                agent_id=agent_id,
                consistency_score=consistency_score,
                goal_alignment_scores=alignment_scores,
                inconsistent_actions=inconsistent_actions,
                dominant_goal_patterns=pattern_strengths,
                temporal_consistency=temporal_consistency
            )
            
            logger.info(f"Goal consistency for {agent_id}: {consistency_score:.3f} "
                       f"(time={processing_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error measuring goal consistency for {agent_id}: {str(e)}")
            return GoalConsistencyResult(
                agent_id=agent_id,
                consistency_score=0.0,
                goal_alignment_scores={},
                inconsistent_actions=[],
                dominant_goal_patterns=[],
                temporal_consistency=0.0
            )
    
    def batch_measure_consistency(self, agents_data: List[Dict[str, Any]]) -> List[GoalConsistencyResult]:
        """Measure goal consistency for multiple agents"""
        results = []
        start_time = time.time()
        
        for agent_data in agents_data:
            result = self.measure_consistency(agent_data)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(agents_data) if agents_data else 0
        
        logger.info(f"Batch goal consistency measurement completed: {len(agents_data)} agents, "
                   f"avg {avg_time:.2f}ms per agent")
        
        return results
    
    def get_consistency_statistics(self, results: List[GoalConsistencyResult]) -> Dict[str, Any]:
        """Calculate statistics from consistency measurement results"""
        if not results:
            return {}
        
        consistent_agents = [r for r in results if r.is_consistent(self.consistency_threshold)]
        
        return {
            "total_agents": len(results),
            "consistent_agents": len(consistent_agents),
            "consistency_rate": len(consistent_agents) / len(results),
            "avg_consistency_score": np.mean([r.consistency_score for r in results]),
            "avg_temporal_consistency": np.mean([r.temporal_consistency for r in results]),
            "consistency_distribution": {
                "high": len([r for r in results if r.consistency_score >= 0.8]),
                "medium": len([r for r in results if 0.5 <= r.consistency_score < 0.8]),
                "low": len([r for r in results if r.consistency_score < 0.5])
            }
        }


# Export main classes
__all__ = [
    'GoalConsistencyMeasurement',
    'GoalTracker',
    'ConsistencyAnalyzer', 
    'GoalConsistencyResult'
]


if __name__ == "__main__":
    # Example usage and testing
    consistency_measurer = GoalConsistencyMeasurement()
    
    # Test with sample agent data
    sample_agent = {
        "agent_id": "sample_agent_001",
        "goals": [
            {
                "goal_type": "management",
                "relevance_score": 0.8,
                "description": "Lead team projects effectively"
            }
        ],
        "goal_description": "I want to organize and coordinate team activities to achieve our objectives",
        "action_history": [
            {"action": "plan_event", "timestamp": time.time() - 3600, "success": True},
            {"action": "coordinate_team", "timestamp": time.time() - 3000, "success": True},
            {"action": "delegate_task", "timestamp": time.time() - 2400, "success": True},
            {"action": "socialize", "timestamp": time.time() - 1800, "success": True},  # Inconsistent
            {"action": "manage_resources", "timestamp": time.time() - 1200, "success": True},
            {"action": "evaluate_performance", "timestamp": time.time() - 600, "success": True},
            {"action": "coordinate_team", "timestamp": time.time(), "success": True},
        ]
    }
    
    result = consistency_measurer.measure_consistency(sample_agent)
    print("Goal Consistency Measurement Result:")
    print(result.to_json())