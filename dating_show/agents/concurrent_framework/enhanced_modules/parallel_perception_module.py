"""
File: parallel_perception_module.py
Description: ParallelPerceptionModule - Concurrent sensory processing for enhanced perception
Enhanced PIANO architecture with parallel processing capabilities
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
import threading
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from enum import Enum
import math

from ...modules.base_module import BaseModule
from ...memory_structures.security_utils import SecurityValidator, SecurityError


class SensorType(Enum):
    """Types of sensory inputs for parallel processing."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    SPATIAL = "spatial"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    TEMPORAL = "temporal"


@dataclass
class PerceptionTask:
    """Represents a perception processing task."""
    task_id: str
    sensor_type: SensorType
    input_data: Any
    priority: float
    region: Optional[Tuple[int, int]] = None  # Spatial region for processing
    timestamp: datetime = None
    processing_time: Optional[float] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerceptionResult:
    """Result of perception processing."""
    task_id: str
    sensor_type: SensorType
    processed_data: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class PerceptionMetrics:
    """Metrics for perception processing performance."""
    total_tasks_processed: int = 0
    average_processing_time: float = 0.0
    tasks_by_sensor_type: Dict[SensorType, int] = None
    concurrent_processing_peak: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    def __post_init__(self):
        if self.tasks_by_sensor_type is None:
            self.tasks_by_sensor_type = {}


class ParallelPerceptionModule(BaseModule):
    """
    Enhanced perception module with parallel processing capabilities.
    Processes multiple sensory inputs concurrently while maintaining coherence.
    """
    
    def __init__(self, agent_state, max_workers: int = 4, enable_caching: bool = True,
                 attention_bandwidth: int = 5, vision_radius: int = 3):
        """
        Initialize ParallelPerceptionModule.
        
        Args:
            agent_state: Shared agent state
            max_workers: Maximum number of concurrent processing threads
            enable_caching: Enable perception result caching
            attention_bandwidth: Number of events that can be processed simultaneously
            vision_radius: Spatial vision radius for environmental perception
        """
        super().__init__(agent_state)
        
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.attention_bandwidth = attention_bandwidth
        self.vision_radius = vision_radius
        
        # Processing components
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Perception")
        self.active_tasks: Dict[str, PerceptionTask] = {}
        self.task_futures: Dict[str, Future] = {}
        
        # Caching system
        self.perception_cache: Dict[str, PerceptionResult] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=5)  # Cache time-to-live
        
        # Attention and prioritization
        self.attention_weights: Dict[SensorType, float] = {
            SensorType.SOCIAL: 1.0,
            SensorType.VISUAL: 0.8,
            SensorType.ENVIRONMENTAL: 0.6,
            SensorType.SPATIAL: 0.4,
            SensorType.AUDITORY: 0.7,
            SensorType.TEMPORAL: 0.3
        }
        
        # Processing queues by priority
        self.high_priority_queue: List[PerceptionTask] = []
        self.normal_priority_queue: List[PerceptionTask] = []
        self.background_queue: List[PerceptionTask] = []
        
        # Metrics and monitoring
        self.metrics = PerceptionMetrics()
        self.processing_history: List[PerceptionResult] = []
        
        # Synchronization
        self._queue_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()
        
        # Background processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="PerceptionProcessor",
            daemon=True
        )
        self._processing_thread.start()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def run(self) -> List[PerceptionResult]:
        """
        Main perception processing run method.
        Processes environmental input and returns perception results.
        
        Returns:
            List of perception results from current processing cycle
        """
        try:
            # Get current environmental data
            environmental_data = self._gather_environmental_data()
            
            # Create perception tasks
            perception_tasks = self._create_perception_tasks(environmental_data)
            
            # Process tasks in parallel
            results = self._process_tasks_parallel(perception_tasks)
            
            # Filter and prioritize results based on attention
            filtered_results = self._apply_attention_filter(results)
            
            # Store results in agent memory
            self._store_perception_results(filtered_results)
            
            # Update metrics
            self._update_metrics(filtered_results)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in perception processing: {e}")
            return []
    
    def _gather_environmental_data(self) -> Dict[str, Any]:
        """Gather environmental data from agent state and surroundings."""
        environmental_data = {}
        
        try:
            # Get spatial information
            if hasattr(self.agent_state, 'curr_tile') and hasattr(self.agent_state, 'maze'):
                maze = self.agent_state.maze
                curr_tile = self.agent_state.curr_tile
                
                # Get nearby tiles within vision radius
                nearby_tiles = self._get_nearby_tiles(maze, curr_tile, self.vision_radius)
                environmental_data['spatial'] = {
                    'current_location': curr_tile,
                    'nearby_tiles': nearby_tiles,
                    'vision_radius': self.vision_radius
                }
                
                # Get events in nearby areas
                nearby_events = self._get_nearby_events(maze, nearby_tiles)
                environmental_data['events'] = nearby_events
                
                # Get social context
                social_context = self._get_social_context(nearby_events)
                environmental_data['social'] = social_context
            
            # Get temporal context
            environmental_data['temporal'] = {
                'current_time': datetime.now(),
                'time_of_day': getattr(self.agent_state, 'curr_time', None)
            }
            
            # Get agent's current state
            environmental_data['agent_state'] = {
                'current_action': getattr(self.agent_state, 'act_description', None),
                'emotional_state': getattr(self.agent_state, 'emotional_state', None),
                'goals': getattr(self.agent_state, 'daily_plan', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error gathering environmental data: {e}")
        
        return environmental_data
    
    def _get_nearby_tiles(self, maze, curr_tile: Tuple[int, int], radius: int) -> List[Dict[str, Any]]:
        """Get tiles within the specified radius."""
        nearby_tiles = []
        
        try:
            x, y = curr_tile
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    tile_x, tile_y = x + dx, y + dy
                    
                    # Check if tile is within maze bounds
                    if hasattr(maze, 'access_tile'):
                        tile_info = maze.access_tile((tile_x, tile_y))
                        if tile_info:
                            distance = math.sqrt(dx*dx + dy*dy)
                            nearby_tiles.append({
                                'coordinates': (tile_x, tile_y),
                                'distance': distance,
                                'info': tile_info
                            })
            
            # Sort by distance
            nearby_tiles.sort(key=lambda t: t['distance'])
            
        except Exception as e:
            self.logger.error(f"Error getting nearby tiles: {e}")
        
        return nearby_tiles
    
    def _get_nearby_events(self, maze, nearby_tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract events from nearby tiles."""
        events = []
        
        try:
            for tile_info in nearby_tiles:
                tile_data = tile_info.get('info', {})
                if tile_data.get('events'):
                    for event in tile_data['events']:
                        events.append({
                            'event': event,
                            'location': tile_info['coordinates'],
                            'distance': tile_info['distance']
                        })
        
        except Exception as e:
            self.logger.error(f"Error getting nearby events: {e}")
        
        return events
    
    def _get_social_context(self, nearby_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract social context from events."""
        social_context = {
            'other_agents': set(),
            'social_interactions': [],
            'conversations': []
        }
        
        try:
            for event_data in nearby_events:
                event = event_data['event']
                if len(event) >= 1:  # (subject, predicate, object, description)
                    subject = event[0]
                    
                    # Check if this involves another agent
                    if ':' in subject and 'agent' in subject.lower():
                        agent_name = subject.split(':')[-1]
                        social_context['other_agents'].add(agent_name)
                        
                        # Check for social interactions
                        if len(event) >= 2 and event[1] in ['chat with', 'talk to', 'interact with']:
                            social_context['social_interactions'].append({
                                'type': 'conversation',
                                'participants': [agent_name],
                                'location': event_data['location'],
                                'distance': event_data['distance']
                            })
        
        except Exception as e:
            self.logger.error(f"Error extracting social context: {e}")
        
        return social_context
    
    def _create_perception_tasks(self, environmental_data: Dict[str, Any]) -> List[PerceptionTask]:
        """Create perception tasks from environmental data."""
        tasks = []
        task_counter = 0
        
        try:
            # Visual perception tasks
            if 'spatial' in environmental_data:
                task_counter += 1
                visual_task = PerceptionTask(
                    task_id=f"visual_{task_counter}",
                    sensor_type=SensorType.VISUAL,
                    input_data=environmental_data['spatial'],
                    priority=self.attention_weights[SensorType.VISUAL]
                )
                tasks.append(visual_task)
            
            # Social perception tasks
            if 'social' in environmental_data and environmental_data['social']['other_agents']:
                task_counter += 1
                social_task = PerceptionTask(
                    task_id=f"social_{task_counter}",
                    sensor_type=SensorType.SOCIAL,
                    input_data=environmental_data['social'],
                    priority=self.attention_weights[SensorType.SOCIAL]
                )
                tasks.append(social_task)
            
            # Environmental perception tasks
            if 'events' in environmental_data:
                task_counter += 1
                env_task = PerceptionTask(
                    task_id=f"environmental_{task_counter}",
                    sensor_type=SensorType.ENVIRONMENTAL,
                    input_data=environmental_data['events'],
                    priority=self.attention_weights[SensorType.ENVIRONMENTAL]
                )
                tasks.append(env_task)
            
            # Temporal perception tasks
            if 'temporal' in environmental_data:
                task_counter += 1
                temporal_task = PerceptionTask(
                    task_id=f"temporal_{task_counter}",
                    sensor_type=SensorType.TEMPORAL,
                    input_data=environmental_data['temporal'],
                    priority=self.attention_weights[SensorType.TEMPORAL]
                )
                tasks.append(temporal_task)
            
            # Spatial perception tasks  
            if 'spatial' in environmental_data:
                task_counter += 1
                spatial_task = PerceptionTask(
                    task_id=f"spatial_{task_counter}",
                    sensor_type=SensorType.SPATIAL,
                    input_data=environmental_data['spatial'],
                    priority=self.attention_weights[SensorType.SPATIAL]
                )
                tasks.append(spatial_task)
        
        except Exception as e:
            self.logger.error(f"Error creating perception tasks: {e}")
        
        return tasks
    
    def _process_tasks_parallel(self, tasks: List[PerceptionTask]) -> List[PerceptionResult]:
        """Process perception tasks in parallel."""
        if not tasks:
            return []
        
        results = []
        
        try:
            # Submit tasks to thread pool
            future_to_task = {}
            for task in tasks[:self.attention_bandwidth]:  # Limit by attention bandwidth
                
                # Check cache first
                if self.enable_caching:
                    cached_result = self._check_cache(task)
                    if cached_result:
                        results.append(cached_result)
                        continue
                
                # Submit for processing
                future = self.thread_pool.submit(self._process_single_task, task)
                future_to_task[future] = task
                self.task_futures[task.task_id] = future
            
            # Collect results
            for future in as_completed(future_to_task.keys(), timeout=10.0):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Cache result
                        if self.enable_caching:
                            self._cache_result(result)
                        
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {e}")
                finally:
                    # Clean up
                    self.task_futures.pop(task.task_id, None)
        
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
        
        return results
    
    def _process_single_task(self, task: PerceptionTask) -> Optional[PerceptionResult]:
        """Process a single perception task."""
        start_time = time.time()
        
        try:
            processed_data = None
            confidence = 0.0
            metadata = {}
            
            if task.sensor_type == SensorType.VISUAL:
                processed_data, confidence = self._process_visual_input(task.input_data)
                metadata = {'processing_type': 'visual_scene_analysis'}
            
            elif task.sensor_type == SensorType.SOCIAL:
                processed_data, confidence = self._process_social_input(task.input_data)
                metadata = {'processing_type': 'social_context_analysis'}
            
            elif task.sensor_type == SensorType.ENVIRONMENTAL:
                processed_data, confidence = self._process_environmental_input(task.input_data)
                metadata = {'processing_type': 'environmental_event_analysis'}
            
            elif task.sensor_type == SensorType.SPATIAL:
                processed_data, confidence = self._process_spatial_input(task.input_data)
                metadata = {'processing_type': 'spatial_mapping'}
            
            elif task.sensor_type == SensorType.TEMPORAL:
                processed_data, confidence = self._process_temporal_input(task.input_data)
                metadata = {'processing_type': 'temporal_context'}
            
            processing_time = time.time() - start_time
            task.processing_time = processing_time
            task.result = processed_data
            
            return PerceptionResult(
                task_id=task.task_id,
                sensor_type=task.sensor_type,
                processed_data=processed_data,
                confidence=confidence,
                processing_time=processing_time,
                metadata=metadata,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            return None
    
    def _process_visual_input(self, visual_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process visual sensory input."""
        try:
            # Analyze spatial layout and objects
            current_location = visual_data.get('current_location')
            nearby_tiles = visual_data.get('nearby_tiles', [])
            
            # Build visual scene representation
            scene_objects = []
            scene_layout = {}
            
            for tile in nearby_tiles[:5]:  # Limit to closest tiles
                tile_info = tile.get('info', {})
                if tile_info.get('game_object'):
                    scene_objects.append({
                        'object': tile_info['game_object'],
                        'location': tile['coordinates'],
                        'distance': tile['distance']
                    })
                
                # Build layout map
                scene_layout[tile['coordinates']] = {
                    'world': tile_info.get('world'),
                    'sector': tile_info.get('sector'),
                    'arena': tile_info.get('arena')
                }
            
            processed_visual = {
                'current_location': current_location,
                'visible_objects': scene_objects,
                'scene_layout': scene_layout,
                'field_of_view': len(nearby_tiles)
            }
            
            confidence = min(1.0, len(scene_objects) / 3.0)  # Higher confidence with more objects
            return processed_visual, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing visual input: {e}")
            return {}, 0.0
    
    def _process_social_input(self, social_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process social sensory input."""
        try:
            other_agents = social_data.get('other_agents', set())
            interactions = social_data.get('social_interactions', [])
            
            # Analyze social context
            social_situation = {
                'present_agents': list(other_agents),
                'interaction_opportunities': len(interactions),
                'social_density': len(other_agents),
                'active_interactions': interactions
            }
            
            # Calculate social importance
            confidence = min(1.0, (len(other_agents) + len(interactions)) / 4.0)
            
            return social_situation, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing social input: {e}")
            return {}, 0.0
    
    def _process_environmental_input(self, event_data: List[Dict[str, Any]]) -> Tuple[Any, float]:
        """Process environmental event input."""
        try:
            # Categorize and analyze events
            significant_events = []
            event_categories = {}
            
            for event_info in event_data[:10]:  # Limit processing
                event = event_info['event']
                if len(event) >= 4:
                    subject, predicate, obj, description = event[:4]
                    
                    # Categorize event
                    if 'chat' in predicate or 'talk' in predicate:
                        category = 'social'
                    elif 'move' in predicate or 'go' in predicate:
                        category = 'movement'
                    elif 'eat' in predicate or 'drink' in predicate:
                        category = 'consumption'
                    else:
                        category = 'general'
                    
                    event_categories[category] = event_categories.get(category, 0) + 1
                    
                    significant_events.append({
                        'description': description,
                        'category': category,
                        'location': event_info['location'],
                        'distance': event_info['distance']
                    })
            
            environmental_analysis = {
                'significant_events': significant_events,
                'event_categories': event_categories,
                'activity_level': len(significant_events),
                'most_common_activity': max(event_categories.items(), key=lambda x: x[1])[0] if event_categories else 'none'
            }
            
            confidence = min(1.0, len(significant_events) / 5.0)
            return environmental_analysis, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing environmental input: {e}")
            return {}, 0.0
    
    def _process_spatial_input(self, spatial_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process spatial sensory input."""
        try:
            current_location = spatial_data.get('current_location')
            nearby_tiles = spatial_data.get('nearby_tiles', [])
            
            # Build spatial map
            spatial_map = {
                'current_position': current_location,
                'accessible_areas': [],
                'notable_locations': [],
                'spatial_relationships': {}
            }
            
            for tile in nearby_tiles:
                tile_info = tile.get('info', {})
                coords = tile['coordinates']
                
                # Identify notable locations
                if tile_info.get('arena'):
                    spatial_map['notable_locations'].append({
                        'type': 'arena',
                        'name': tile_info['arena'],
                        'coordinates': coords,
                        'distance': tile['distance']
                    })
                
                # Build accessibility map
                if not tile_info.get('collision', False):
                    spatial_map['accessible_areas'].append(coords)
                
                # Record spatial relationships
                direction = self._get_direction(current_location, coords)
                spatial_map['spatial_relationships'][coords] = {
                    'direction': direction,
                    'distance': tile['distance']
                }
            
            confidence = min(1.0, len(nearby_tiles) / 8.0)  # Higher confidence with more spatial info
            return spatial_map, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing spatial input: {e}")
            return {}, 0.0
    
    def _process_temporal_input(self, temporal_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Process temporal sensory input."""
        try:
            current_time = temporal_data.get('current_time', datetime.now())
            time_of_day = temporal_data.get('time_of_day')
            
            # Analyze temporal context
            temporal_context = {
                'current_timestamp': current_time,
                'time_of_day': time_of_day,
                'hour': current_time.hour,
                'time_period': self._classify_time_period(current_time.hour),
                'temporal_patterns': self._identify_temporal_patterns()
            }
            
            confidence = 0.8  # Temporal processing is generally reliable
            return temporal_context, confidence
            
        except Exception as e:
            self.logger.error(f"Error processing temporal input: {e}")
            return {}, 0.0
    
    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Get cardinal direction from one position to another."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy):
            return 'east' if dx > 0 else 'west'
        else:
            return 'south' if dy > 0 else 'north'
    
    def _classify_time_period(self, hour: int) -> str:
        """Classify hour into time periods."""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _identify_temporal_patterns(self) -> Dict[str, Any]:
        """Identify temporal patterns from processing history."""
        patterns = {
            'recent_activity': [],
            'time_since_last_major_event': None,
            'processing_frequency': 0.0
        }
        
        try:
            # Analyze recent processing history
            recent_results = [r for r in self.processing_history[-10:] 
                           if (datetime.now() - r.timestamp).total_seconds() < 300]  # Last 5 minutes
            
            patterns['recent_activity'] = [r.sensor_type.value for r in recent_results]
            patterns['processing_frequency'] = len(recent_results) / 5.0  # Per minute
            
        except Exception as e:
            self.logger.error(f"Error identifying temporal patterns: {e}")
        
        return patterns
    
    def _apply_attention_filter(self, results: List[PerceptionResult]) -> List[PerceptionResult]:
        """Apply attention filtering based on importance and capacity."""
        if len(results) <= self.attention_bandwidth:
            return results
        
        # Sort by priority (sensor type weight * confidence)
        prioritized_results = sorted(
            results,
            key=lambda r: self.attention_weights.get(r.sensor_type, 0.5) * r.confidence,
            reverse=True
        )
        
        return prioritized_results[:self.attention_bandwidth]
    
    def _store_perception_results(self, results: List[PerceptionResult]) -> None:
        """Store perception results in agent memory."""
        try:
            for result in results:
                # Store in processing history
                self.processing_history.append(result)
                
                # Keep history bounded
                if len(self.processing_history) > 100:
                    self.processing_history = self.processing_history[-50:]
                
                # Store in agent's episodic memory if available
                if hasattr(self.agent_state, 'episodic_memory'):
                    memory_content = f"Perceived {result.sensor_type.value}: {result.processed_data}"
                    self.agent_state.episodic_memory.add_event(
                        content=memory_content,
                        event_type="perception",
                        importance=result.confidence,
                        metadata={
                            'sensor_type': result.sensor_type.value,
                            'processing_time': result.processing_time,
                            'confidence': result.confidence
                        }
                    )
        
        except Exception as e:
            self.logger.error(f"Error storing perception results: {e}")
    
    def _check_cache(self, task: PerceptionTask) -> Optional[PerceptionResult]:
        """Check if a similar perception result is cached."""
        if not self.enable_caching:
            return None
        
        try:
            with self._cache_lock:
                # Generate cache key based on task input
                cache_key = self._generate_cache_key(task)
                
                if cache_key in self.perception_cache:
                    cached_result = self.perception_cache[cache_key]
                    cache_time = self.cache_timestamps[cache_key]
                    
                    # Check if cache is still valid
                    if datetime.now() - cache_time < self.cache_ttl:
                        self.logger.debug(f"Cache hit for task {task.task_id}")
                        return cached_result
                    else:
                        # Remove expired cache entry
                        del self.perception_cache[cache_key]
                        del self.cache_timestamps[cache_key]
        
        except Exception as e:
            self.logger.error(f"Error checking cache: {e}")
        
        return None
    
    def _cache_result(self, result: PerceptionResult) -> None:
        """Cache a perception result."""
        if not self.enable_caching:
            return
        
        try:
            with self._cache_lock:
                # Generate cache key
                cache_key = f"{result.sensor_type.value}_{hash(str(result.processed_data))}"
                
                self.perception_cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now()
                
                # Limit cache size
                if len(self.perception_cache) > 100:
                    oldest_key = min(self.cache_timestamps.keys(), 
                                   key=lambda k: self.cache_timestamps[k])
                    del self.perception_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]
        
        except Exception as e:
            self.logger.error(f"Error caching result: {e}")
    
    def _generate_cache_key(self, task: PerceptionTask) -> str:
        """Generate cache key for a perception task."""
        return f"{task.sensor_type.value}_{hash(str(task.input_data))}"
    
    def _update_metrics(self, results: List[PerceptionResult]) -> None:
        """Update processing metrics."""
        try:
            self.metrics.total_tasks_processed += len(results)
            
            if results:
                # Update average processing time
                total_time = sum(r.processing_time for r in results)
                avg_time = total_time / len(results)
                
                # Running average
                n = self.metrics.total_tasks_processed
                self.metrics.average_processing_time = (
                    (self.metrics.average_processing_time * (n - len(results)) + total_time) / n
                )
                
                # Update sensor type counts
                for result in results:
                    sensor_type = result.sensor_type
                    self.metrics.tasks_by_sensor_type[sensor_type] = (
                        self.metrics.tasks_by_sensor_type.get(sensor_type, 0) + 1
                    )
                
                # Update concurrent processing peak
                current_concurrent = len(self.task_futures)
                if current_concurrent > self.metrics.concurrent_processing_peak:
                    self.metrics.concurrent_processing_peak = current_concurrent
        
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _processing_loop(self) -> None:
        """Background processing loop for handling queued tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up completed futures
                completed_futures = [
                    task_id for task_id, future in self.task_futures.items()
                    if future.done()
                ]
                
                for task_id in completed_futures:
                    self.task_futures.pop(task_id, None)
                
                # Clean up expired cache entries
                if self.enable_caching:
                    self._cleanup_cache()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            with self._cache_lock:
                current_time = datetime.now()
                expired_keys = [
                    key for key, timestamp in self.cache_timestamps.items()
                    if current_time - timestamp > self.cache_ttl
                ]
                
                for key in expired_keys:
                    self.perception_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
        
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
    
    def get_perception_metrics(self) -> PerceptionMetrics:
        """Get current perception processing metrics."""
        return self.metrics
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            'active_tasks': len(self.active_tasks),
            'pending_futures': len(self.task_futures),
            'cache_size': len(self.perception_cache),
            'processing_history_length': len(self.processing_history),
            'attention_bandwidth': self.attention_bandwidth,
            'vision_radius': self.vision_radius,
            'caching_enabled': self.enable_caching
        }
    
    def shutdown(self) -> None:
        """Shutdown the perception module."""
        self.logger.info("Shutting down ParallelPerceptionModule...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel active tasks
        for future in self.task_futures.values():
            future.cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=10.0)
        
        # Wait for processing thread
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
        
        self.logger.info("ParallelPerceptionModule shutdown complete")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ParallelPerceptionModule(workers={self.max_workers}, "
                f"attention_bandwidth={self.attention_bandwidth}, "
                f"active_tasks={len(self.active_tasks)})")