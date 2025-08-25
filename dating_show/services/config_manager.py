"""
Configuration Management System
Handles loading and management of simulation configuration
Supports YAML configuration with environment variable overrides
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class StepGenerationSettings:
    """Step generation configuration"""
    strategy: str = "hybrid"
    fallback_enabled: bool = True
    cache_templates: bool = True
    max_retries: int = 3
    generation_timeout: float = 30.0
    generation_order: list = None
    
    def __post_init__(self):
        if self.generation_order is None:
            self.generation_order = ["previous_step", "base_simulation", "minimal_fallback"]

@dataclass
class DatingShowSettings:
    """Dating show specific configuration"""
    context_enhancement: bool = True
    emoji_pool: list = None
    villa_activities: list = None
    position_evolution: dict = None
    
    def __post_init__(self):
        if self.emoji_pool is None:
            self.emoji_pool = ["💕", "🌹", "💬", "😊", "🥰", "✨", "🌟"]
        if self.villa_activities is None:
            self.villa_activities = [
                "socializing with other contestants",
                "exploring the villa grounds", 
                "having conversations by the pool"
            ]
        if self.position_evolution is None:
            self.position_evolution = {
                "enabled": True,
                "max_movement": 5,
                "boundaries": {"min_x": 15, "max_x": 90, "min_y": 15, "max_y": 75},
                "description_update_chance": 0.3
            }

@dataclass
class StorageSettings:
    """Storage configuration"""
    primary_path: str = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
    backup_path: str = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage"
    temp_storage: str = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/temp_storage"
    patterns: dict = None
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = {
                "environment": "environment/{step}.json",
                "movement": "movement/{step}.json",
                "meta": "reverie/meta.json"
            }

@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    step_generation: StepGenerationSettings = None
    dating_show: DatingShowSettings = None
    storage: StorageSettings = None
    real_time: dict = None
    agent_processing: dict = None
    error_handling: dict = None
    performance: dict = None
    debug: dict = None
    compatibility: dict = None
    
    def __post_init__(self):
        if self.step_generation is None:
            self.step_generation = StepGenerationSettings()
        if self.dating_show is None:
            self.dating_show = DatingShowSettings()
        if self.storage is None:
            self.storage = StorageSettings()

class ConfigManager:
    """
    Configuration manager for simulation orchestration
    Handles loading, validation, and environment variable overrides
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        self._config: Optional[SimulationConfig] = None
        self._raw_config: Dict[str, Any] = {}
        
    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve configuration file path"""
        if config_path:
            return Path(config_path)
        
        # Try multiple locations
        candidates = [
            Path("dating_show/config/simulation_config.yaml"),
            Path("config/simulation_config.yaml"),
            Path("simulation_config.yaml"),
            Path(__file__).parent.parent / "config" / "simulation_config.yaml"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Return first candidate as default (will create if needed)
        return candidates[0]
    
    def load_config(self) -> SimulationConfig:
        """Load configuration from file with environment overrides"""
        try:
            # Load YAML configuration
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self._raw_config = yaml.safe_load(f) or {}
                logger.info(f"✅ Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"⚠️ Configuration file not found: {self.config_path}")
                self._raw_config = {}
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Create structured configuration
            self._config = self._create_config_object()
            
            return self._config
            
        except Exception as e:
            logger.error(f"🚨 Configuration loading error: {e}")
            # Return default configuration on error
            return SimulationConfig()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration"""
        env_mappings = {
            'DATING_SHOW_STORAGE_PATH': ['storage', 'primary_path'],
            'DATING_SHOW_STRATEGY': ['step_generation', 'strategy'],
            'DATING_SHOW_DEBUG': ['debug', 'debug_logging'],
            'DATING_SHOW_ASYNC': ['performance', 'async_operations'],
            'DATING_SHOW_WEBSOCKET': ['real_time', 'websocket_enabled'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_config(config_path, self._parse_env_value(env_value))
    
    def _set_nested_config(self, path: list, value: Any):
        """Set nested configuration value"""
        current = self._raw_config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _parse_env_value(self, value: str) -> Union[str, bool, int, float]:
        """Parse environment variable value to appropriate type"""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def _create_config_object(self) -> SimulationConfig:
        """Create structured configuration object from raw config"""
        try:
            # Extract step generation settings
            step_gen_data = self._raw_config.get('step_generation', {})
            step_generation = StepGenerationSettings(
                strategy=step_gen_data.get('strategy', 'hybrid'),
                fallback_enabled=step_gen_data.get('fallback_enabled', True),
                cache_templates=step_gen_data.get('cache_templates', True),
                max_retries=step_gen_data.get('max_retries', 3),
                generation_timeout=step_gen_data.get('generation_timeout', 30.0),
                generation_order=step_gen_data.get('generation_order', ["previous_step", "base_simulation", "minimal_fallback"])
            )
            
            # Extract dating show settings
            dating_data = self._raw_config.get('dating_show', {})
            dating_show = DatingShowSettings(
                context_enhancement=dating_data.get('context_enhancement', True),
                emoji_pool=dating_data.get('emoji_pool'),
                villa_activities=dating_data.get('villa_activities'),
                position_evolution=dating_data.get('position_evolution')
            )
            
            # Extract storage settings
            storage_data = self._raw_config.get('storage', {})
            storage = StorageSettings(
                primary_path=storage_data.get('primary_path', StorageSettings().primary_path),
                backup_path=storage_data.get('backup_path', StorageSettings().backup_path),
                temp_storage=storage_data.get('temp_storage', StorageSettings().temp_storage),
                patterns=storage_data.get('patterns')
            )
            
            # Create complete configuration
            return SimulationConfig(
                step_generation=step_generation,
                dating_show=dating_show,
                storage=storage,
                real_time=self._raw_config.get('real_time', {}),
                agent_processing=self._raw_config.get('agent_processing', {}),
                error_handling=self._raw_config.get('error_handling', {}),
                performance=self._raw_config.get('performance', {}),
                debug=self._raw_config.get('debug', {}),
                compatibility=self._raw_config.get('compatibility', {})
            )
            
        except Exception as e:
            logger.error(f"Configuration parsing error: {e}")
            return SimulationConfig()
    
    def get_config(self) -> SimulationConfig:
        """Get current configuration (load if not already loaded)"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> SimulationConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load_config()
    
    def save_config(self, config: Optional[SimulationConfig] = None):
        """Save current configuration to file"""
        try:
            config = config or self._config
            if not config:
                logger.warning("No configuration to save")
                return
            
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and save
            config_dict = asdict(config)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Configuration save error: {e}")
    
    # Utility methods for specific configuration access
    
    def get_storage_path(self) -> str:
        """Get primary storage path"""
        config = self.get_config()
        return config.storage.primary_path
    
    def get_step_generation_strategy(self) -> str:
        """Get step generation strategy"""
        config = self.get_config()
        return config.step_generation.strategy
    
    def is_websocket_enabled(self) -> bool:
        """Check if WebSocket broadcasting is enabled"""
        config = self.get_config()
        return config.real_time.get('websocket_enabled', False)
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        config = self.get_config()
        return config.debug.get('debug_logging', False)
    
    def get_emoji_pool(self) -> list:
        """Get dating show emoji pool"""
        config = self.get_config()
        return config.dating_show.emoji_pool
    
    def get_villa_activities(self) -> list:
        """Get villa activities list"""
        config = self.get_config()
        return config.dating_show.villa_activities


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get singleton configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager

def get_simulation_config() -> SimulationConfig:
    """Get current simulation configuration"""
    return get_config_manager().get_config()