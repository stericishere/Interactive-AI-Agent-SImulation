"""
Specialization System for Enhanced PIANO Architecture
Week 3: Specialization System Implementation

This module implements role detection, skill development, and professional identity
formation for AI agents in the dating show simulation.
"""

__version__ = "1.0.0"
__author__ = "Enhanced PIANO Architecture Team"

from .role_detector import RoleDetector, ActionPatternAnalyzer, ProfessionalBehaviorClassifier, RoleClassificationResult

__all__ = [
    'RoleDetector',
    'ActionPatternAnalyzer', 
    'ProfessionalBehaviorClassifier',
    'RoleClassificationResult'
]