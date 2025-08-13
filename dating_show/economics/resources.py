"""
Resource Management System for Enhanced PIANO Architecture
Phase 3: Advanced Features - Week 10: Economic Systems

This module implements a comprehensive resource management system that handles
resource tracking, allocation, scarcity management, and trade mechanisms for
multi-agent economic simulation.
"""

import json
import math
import time
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from statistics import mean, median, stdev
import random
import logging

# Set up logging for production monitoring
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources in the economic system"""
    # Basic Resources
    FOOD = "food"
    WATER = "water"
    SHELTER = "shelter"
    ENERGY = "energy"
    
    # Materials
    WOOD = "wood"
    STONE = "stone"
    METAL = "metal"
    FABRIC = "fabric"
    
    # Advanced Resources
    TOOLS = "tools"
    TECHNOLOGY = "technology"
    INFORMATION = "information"
    KNOWLEDGE = "knowledge"
    
    # Social Resources
    REPUTATION = "reputation"
    INFLUENCE = "influence"
    TRUST = "trust"
    SOCIAL_CAPITAL = "social_capital"
    
    # Economic Resources
    CURRENCY = "currency"
    CREDIT = "credit"
    LAND = "land"
    LABOR = "labor"


class ResourceCategory(Enum):
    """Categories of resources for management purposes"""
    CONSUMABLE = "consumable"      # Used up when consumed
    DURABLE = "durable"            # Can be used multiple times
    RENEWABLE = "renewable"        # Regenerates over time
    FINITE = "finite"              # Limited total supply
    ABSTRACT = "abstract"          # Non-physical resources
    SOCIAL = "social"              # Relationship-based resources


class AllocationMethod(Enum):
    """Methods for resource allocation"""
    EQUAL = "equal"                # Equal distribution
    NEED_BASED = "need_based"      # Based on need/priority
    MERIT_BASED = "merit_based"    # Based on contribution/skill
    AUCTION = "auction"            # Highest bidder
    LOTTERY = "lottery"            # Random allocation
    COALITION = "coalition"        # Coalition-based distribution
    MARKET = "market"              # Market-based pricing


@dataclass
class ResourceInstance:
    """Represents a specific instance of a resource"""
    resource_id: str
    resource_type: ResourceType
    category: ResourceCategory
    quantity: float
    quality: float  # 0.0 to 1.0
    durability: float  # 0.0 to 1.0 (for durable resources)
    location: str
    owner_id: Optional[str]
    created_time: float
    last_updated: float
    expiry_time: Optional[float]  # For perishable resources
    maintenance_cost: float  # Cost to maintain per time unit
    trade_value: float  # Current market value
    transferable: bool
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate resource data"""
        self.quantity = max(0.0, self.quantity)
        self.quality = max(0.0, min(1.0, self.quality))
        self.durability = max(0.0, min(1.0, self.durability))
        self.maintenance_cost = max(0.0, self.maintenance_cost)
        self.trade_value = max(0.0, self.trade_value)


@dataclass
class ResourceNeed:
    """Represents an agent's need for a resource"""
    agent_id: str
    resource_type: ResourceType
    quantity_needed: float
    priority: float  # 0.0 to 1.0
    urgency: float  # 0.0 to 1.0
    max_price: float  # Maximum willing to pay
    deadline: Optional[float]
    alternative_resources: List[ResourceType]
    partial_acceptable: bool
    created_time: float
    
    def __post_init__(self):
        """Validate need data"""
        self.quantity_needed = max(0.0, self.quantity_needed)
        self.priority = max(0.0, min(1.0, self.priority))
        self.urgency = max(0.0, min(1.0, self.urgency))
        self.max_price = max(0.0, self.max_price)


@dataclass
class TradeOffer:
    """Represents a trade offer between agents"""
    offer_id: str
    proposer_id: str
    recipient_id: Optional[str]  # None for public offers
    offered_resources: Dict[str, float]  # resource_id -> quantity
    requested_resources: Dict[ResourceType, float]  # resource_type -> quantity
    currency_offered: float
    currency_requested: float
    expiry_time: float
    trade_ratio: float  # Value offered / Value requested
    created_time: float
    status: str  # "pending", "accepted", "rejected", "expired", "completed"
    conditions: List[str]
    
    def __post_init__(self):
        """Validate trade offer data"""
        self.currency_offered = max(0.0, self.currency_offered)
        self.currency_requested = max(0.0, self.currency_requested)
        self.trade_ratio = max(0.0, self.trade_ratio)


@dataclass
class Market:
    """Represents a marketplace for resource trading"""
    market_id: str
    name: str
    location: str
    resource_types: Set[ResourceType]
    active_offers: List[str]  # offer IDs
    price_history: Dict[ResourceType, List[Tuple[float, float]]]  # timestamp, price
    trade_volume: Dict[ResourceType, float]
    transaction_fee: float
    operating_hours: Tuple[float, float]  # start_time, end_time (hours of day)
    reputation_required: float  # Minimum reputation to trade
    access_agents: Set[str]  # Agents with access
    created_time: float
    last_activity: float


class ResourceManagementSystem:
    """
    Comprehensive resource management system for multi-agent economic simulation.
    
    Features:
    - Resource tracking and allocation
    - Scarcity management and rationing
    - Multi-agent trading mechanisms
    - Market dynamics and pricing
    - Resource production and consumption
    - Efficiency optimization
    - Economic inequality tracking
    """
    
    def __init__(self, max_agents: int = 1000, max_resources: int = 10000):
        self.max_agents = max_agents
        self.max_resources = max_resources
        
        # Core data structures
        self.agents: Set[str] = set()
        self.resources: Dict[str, ResourceInstance] = {}
        self.agent_inventory: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.agent_needs: Dict[str, List[ResourceNeed]] = defaultdict(list)
        
        # Economic structures
        self.markets: Dict[str, Market] = {}
        self.trade_offers: Dict[str, TradeOffer] = {}
        self.currency_balances: Dict[str, float] = defaultdict(float)
        
        # Resource configuration
        self.resource_config = self._initialize_resource_config()
        self.base_prices: Dict[ResourceType, float] = self._initialize_base_prices()
        self.scarcity_thresholds: Dict[ResourceType, float] = self._initialize_scarcity_thresholds()
        
        # Production and consumption
        self.production_rates: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self.consumption_rates: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        
        # Market dynamics
        self.supply_demand_cache: Dict[ResourceType, Tuple[float, float]] = {}  # supply, demand
        self.price_cache: Dict[ResourceType, float] = {}
        
        # Performance tracking
        self.transaction_history: List[Dict[str, Any]] = []
        self.efficiency_metrics: Dict[str, float] = {}
        
        # Concurrency control
        self.lock = threading.RLock()
        
        # Caching
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        
        # Economic parameters
        self.inflation_rate = 0.02  # Annual inflation rate
        self.interest_rate = 0.05  # Annual interest rate
        self.transaction_fee_rate = 0.01  # 1% transaction fee
        
    def add_agent(
        self, 
        agent_id: str, 
        initial_currency: float = 100.0,
        initial_resources: Optional[Dict[ResourceType, float]] = None
    ) -> bool:
        """Add an agent to the resource management system"""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents.add(agent_id)
            self.currency_balances[agent_id] = initial_currency
            
            # Add initial resources
            if initial_resources:
                for resource_type, quantity in initial_resources.items():
                    self.create_resource(resource_type, quantity, agent_id)
            
            # Initialize default needs
            self._initialize_agent_needs(agent_id)
            
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            # Transfer resources to system (could be distributed to other agents)
            agent_resources = []
            for resource_id, resource in self.resources.items():
                if resource.owner_id == agent_id:
                    agent_resources.append(resource_id)
            
            for resource_id in agent_resources:
                self.resources[resource_id].owner_id = None  # Make ownerless
            
            # Cancel active trades
            active_offers = [offer_id for offer_id, offer in self.trade_offers.items()
                           if offer.proposer_id == agent_id or offer.recipient_id == agent_id]
            
            for offer_id in active_offers:
                self.trade_offers[offer_id].status = "cancelled"
            
            # Remove from system
            self.agents.remove(agent_id)
            self.currency_balances.pop(agent_id, None)
            self.agent_inventory.pop(agent_id, None)
            self.agent_needs.pop(agent_id, None)
            self.production_rates.pop(agent_id, None)
            self.consumption_rates.pop(agent_id, None)
            
            return True
    
    def create_resource(
        self, 
        resource_type: ResourceType, 
        quantity: float, 
        owner_id: Optional[str] = None,
        quality: float = 1.0,
        location: str = "default"
    ) -> str:
        """Create a new resource instance"""
        # Input validation
        if not isinstance(resource_type, ResourceType):
            return ""
        if quantity <= 0.0:
            return ""
        if owner_id and owner_id not in self.agents:
            return ""
        quality = max(0.0, min(1.0, quality))
        
        with self.lock:
            if len(self.resources) >= self.max_resources:
                return ""
            
            resource_id = f"{resource_type.value}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            config = self.resource_config[resource_type]
            
            # Calculate expiry time for perishable resources
            expiry_time = None
            if config["category"] == ResourceCategory.CONSUMABLE and "shelf_life" in config and config["shelf_life"] is not None:
                expiry_time = time.time() + config["shelf_life"]
            
            resource = ResourceInstance(
                resource_id=resource_id,
                resource_type=resource_type,
                category=config["category"],
                quantity=quantity,
                quality=quality,
                durability=config.get("initial_durability", 1.0),
                location=location,
                owner_id=owner_id,
                created_time=time.time(),
                last_updated=time.time(),
                expiry_time=expiry_time,
                maintenance_cost=config.get("maintenance_cost", 0.0),
                trade_value=self.base_prices[resource_type] * quantity * quality,
                transferable=config.get("transferable", True),
                metadata={}
            )
            
            self.resources[resource_id] = resource
            
            # Update agent inventory
            if owner_id:
                self.agent_inventory[owner_id][resource_id] += quantity
            
            self._invalidate_cache()
            return resource_id
    
    def transfer_resource(
        self, 
        resource_id: str, 
        from_agent: str, 
        to_agent: str, 
        quantity: Optional[float] = None
    ) -> bool:
        """Transfer resource between agents"""
        with self.lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources[resource_id]
            
            if resource.owner_id != from_agent:
                return False
            
            if not resource.transferable:
                return False
            
            # Check quantity
            transfer_quantity = quantity if quantity is not None else resource.quantity
            available_quantity = self.agent_inventory[from_agent][resource_id]
            
            if transfer_quantity > available_quantity:
                return False
            
            # Perform transfer
            self.agent_inventory[from_agent][resource_id] -= transfer_quantity
            self.agent_inventory[to_agent][resource_id] += transfer_quantity
            
            # Update resource ownership if fully transferred
            if self.agent_inventory[from_agent][resource_id] <= 0:
                del self.agent_inventory[from_agent][resource_id]
                resource.owner_id = to_agent
            
            resource.last_updated = time.time()
            
            self._record_transaction(from_agent, to_agent, resource_id, transfer_quantity, 0.0, "transfer")
            
            return True
    
    def consume_resource(
        self, 
        agent_id: str, 
        resource_type: ResourceType, 
        quantity: float,
        efficiency: float = 1.0
    ) -> float:
        """Consume a resource and return actual quantity consumed"""
        with self.lock:
            if agent_id not in self.agents:
                return 0.0
            
            # Find available resources of this type
            available_resources = []
            for resource_id, resource in self.resources.items():
                if (resource.resource_type == resource_type and 
                    resource.owner_id == agent_id and
                    self.agent_inventory[agent_id][resource_id] > 0):
                    available_resources.append((resource_id, resource))
            
            # Sort by quality (consume higher quality first)
            available_resources.sort(key=lambda x: x[1].quality, reverse=True)
            
            total_consumed = 0.0
            remaining_needed = quantity
            
            for resource_id, resource in available_resources:
                if remaining_needed <= 0:
                    break
                
                available_quantity = self.agent_inventory[agent_id][resource_id]
                consume_quantity = min(remaining_needed, available_quantity)
                
                # Apply efficiency modifier
                effective_consumption = consume_quantity * efficiency
                
                # Update inventory
                self.agent_inventory[agent_id][resource_id] -= consume_quantity
                if self.agent_inventory[agent_id][resource_id] <= 0:
                    del self.agent_inventory[agent_id][resource_id]
                
                # Update resource quantity
                resource.quantity -= consume_quantity
                if resource.quantity <= 0:
                    del self.resources[resource_id]
                else:
                    resource.last_updated = time.time()
                
                total_consumed += effective_consumption
                remaining_needed -= consume_quantity  # Reduce by actual quantity consumed, not effective
            
            # Update consumption tracking
            self.consumption_rates[agent_id][resource_type] += total_consumed
            
            self._invalidate_cache()
            return total_consumed
    
    def produce_resource(
        self, 
        agent_id: str, 
        resource_type: ResourceType, 
        quantity: float,
        inputs: Optional[Dict[ResourceType, float]] = None,
        efficiency: float = 1.0,
        location: str = "default"
    ) -> str:
        """Produce a new resource using inputs"""
        with self.lock:
            if agent_id not in self.agents:
                return ""
            
            # Check and consume input resources
            if inputs:
                for input_type, input_quantity in inputs.items():
                    consumed = self.consume_resource(agent_id, input_type, input_quantity)
                    if consumed < input_quantity * 0.9:  # Allow 10% shortage
                        # Insufficient inputs, refund what was consumed
                        return ""
            
            # Calculate production efficiency
            actual_quantity = quantity * efficiency
            
            # Create the produced resource
            resource_id = self.create_resource(resource_type, actual_quantity, agent_id, location=location)
            
            # Update production tracking
            self.production_rates[agent_id][resource_type] += actual_quantity
            
            return resource_id
    
    def get_agent_inventory(self, agent_id: str) -> Dict[ResourceType, float]:
        """Get agent's inventory grouped by resource type"""
        if agent_id not in self.agents:
            return {}
        
        inventory = defaultdict(float)
        
        for resource_id, quantity in self.agent_inventory[agent_id].items():
            if resource_id in self.resources:
                resource_type = self.resources[resource_id].resource_type
                inventory[resource_type] += quantity
        
        return dict(inventory)
    
    def get_resource_availability(self, resource_type: ResourceType, location: Optional[str] = None) -> Dict[str, Any]:
        """Get availability information for a resource type"""
        total_quantity = 0.0
        total_value = 0.0
        owner_distribution = defaultdict(float)
        quality_distribution = []
        location_distribution = defaultdict(float)
        
        for resource in self.resources.values():
            if resource.resource_type == resource_type:
                if location is None or resource.location == location:
                    total_quantity += resource.quantity
                    total_value += resource.trade_value
                    quality_distribution.append(resource.quality)
                    location_distribution[resource.location] += resource.quantity
                    
                    if resource.owner_id:
                        owner_distribution[resource.owner_id] += resource.quantity
        
        return {
            'total_quantity': total_quantity,
            'total_value': total_value,
            'average_quality': mean(quality_distribution) if quality_distribution else 0.0,
            'num_owners': len(owner_distribution),
            'owner_distribution': dict(owner_distribution),
            'location_distribution': dict(location_distribution),
            'scarcity_level': self._calculate_scarcity_level(resource_type, total_quantity)
        }
    
    def create_trade_offer(
        self,
        proposer_id: str,
        offered_resources: Dict[str, float],  # resource_id -> quantity
        requested_resources: Dict[ResourceType, float],  # resource_type -> quantity
        currency_offered: float = 0.0,
        currency_requested: float = 0.0,
        recipient_id: Optional[str] = None,
        expiry_hours: float = 24.0
    ) -> str:
        """Create a trade offer"""
        with self.lock:
            if proposer_id not in self.agents:
                return ""
            
            # Validate offered resources
            for resource_id, quantity in offered_resources.items():
                if resource_id not in self.resources:
                    return ""
                
                resource = self.resources[resource_id]
                if resource.owner_id != proposer_id:
                    return ""
                
                available = self.agent_inventory[proposer_id][resource_id]
                if quantity > available:
                    return ""
            
            # Validate currency
            if currency_offered > self.currency_balances[proposer_id]:
                return ""
            
            offer_id = f"offer_{proposer_id}_{int(time.time() * 1000)}"
            
            # Calculate trade ratio
            offered_value = currency_offered
            for resource_id, quantity in offered_resources.items():
                resource = self.resources[resource_id]
                offered_value += resource.trade_value * quantity / resource.quantity
            
            requested_value = currency_requested
            for resource_type, quantity in requested_resources.items():
                requested_value += self.get_current_price(resource_type) * quantity
            
            trade_ratio = offered_value / max(requested_value, 0.01)
            
            offer = TradeOffer(
                offer_id=offer_id,
                proposer_id=proposer_id,
                recipient_id=recipient_id,
                offered_resources=offered_resources.copy(),
                requested_resources=requested_resources.copy(),
                currency_offered=currency_offered,
                currency_requested=currency_requested,
                expiry_time=time.time() + expiry_hours * 3600,
                trade_ratio=trade_ratio,
                created_time=time.time(),
                status="pending",
                conditions=[]
            )
            
            self.trade_offers[offer_id] = offer
            return offer_id
    
    def accept_trade_offer(self, offer_id: str, accepter_id: str) -> bool:
        """Accept a trade offer"""
        with self.lock:
            if offer_id not in self.trade_offers:
                return False
            
            offer = self.trade_offers[offer_id]
            
            if offer.status != "pending":
                return False
            
            if offer.expiry_time < time.time():
                offer.status = "expired"
                return False
            
            if offer.recipient_id and offer.recipient_id != accepter_id:
                return False
            
            if accepter_id == offer.proposer_id:
                return False
            
            # Validate accepter has requested resources
            for resource_type, quantity in offer.requested_resources.items():
                available = self.get_agent_inventory(accepter_id).get(resource_type, 0.0)
                if available < quantity:
                    return False
            
            if offer.currency_requested > self.currency_balances[accepter_id]:
                return False
            
            # Execute the trade
            success = self._execute_trade(offer, accepter_id)
            
            if success:
                offer.status = "completed"
                self._record_transaction(
                    offer.proposer_id, 
                    accepter_id, 
                    offer_id, 
                    1.0,  # Quantity = 1 trade
                    offer.currency_offered + offer.currency_requested,
                    "trade"
                )
            else:
                offer.status = "failed"
            
            return success
    
    def get_current_price(self, resource_type: ResourceType) -> float:
        """Get current market price for a resource type"""
        if self._is_cache_valid() and resource_type in self.price_cache:
            return self.price_cache[resource_type]
        
        # Calculate price based on supply and demand
        supply, demand = self._calculate_supply_demand(resource_type)
        base_price = self.base_prices[resource_type]
        
        # Price increases with demand/supply ratio
        if supply > 0:
            price_multiplier = (demand / supply) ** 0.5
        else:
            price_multiplier = 2.0  # High price when no supply
        
        # Apply scarcity modifier
        scarcity_level = self._get_scarcity_level(resource_type)
        scarcity_multiplier = 1.0 + scarcity_level * 2.0
        
        # Apply inflation
        time_factor = (time.time() - 1640995200) / (365.25 * 24 * 3600)  # Years since 2022
        inflation_multiplier = (1.0 + self.inflation_rate) ** time_factor
        
        current_price = base_price * price_multiplier * scarcity_multiplier * inflation_multiplier
        
        # Cache the result
        self.price_cache[resource_type] = current_price
        
        return current_price
    
    def create_market(
        self,
        market_id: str,
        name: str,
        location: str,
        resource_types: Set[ResourceType],
        transaction_fee: float = 0.01,
        reputation_required: float = 0.0
    ) -> bool:
        """Create a new marketplace"""
        with self.lock:
            if market_id in self.markets:
                return False
            
            market = Market(
                market_id=market_id,
                name=name,
                location=location,
                resource_types=resource_types,
                active_offers=[],
                price_history=defaultdict(list),
                trade_volume=defaultdict(float),
                transaction_fee=transaction_fee,
                operating_hours=(0.0, 24.0),  # 24/7 by default
                reputation_required=reputation_required,
                access_agents=set(),
                created_time=time.time(),
                last_activity=time.time()
            )
            
            self.markets[market_id] = market
            return True
    
    def add_need(
        self,
        agent_id: str,
        resource_type: ResourceType,
        quantity: float,
        priority: float = 0.5,
        urgency: float = 0.5,
        max_price: float = float('inf'),
        deadline: Optional[float] = None
    ) -> bool:
        """Add a resource need for an agent"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            need = ResourceNeed(
                agent_id=agent_id,
                resource_type=resource_type,
                quantity_needed=quantity,
                priority=priority,
                urgency=urgency,
                max_price=max_price,
                deadline=deadline,
                alternative_resources=[],
                partial_acceptable=True,
                created_time=time.time()
            )
            
            self.agent_needs[agent_id].append(need)
            return True
    
    def allocate_resources(
        self,
        resource_type: ResourceType,
        total_quantity: float,
        agents: List[str],
        method: AllocationMethod = AllocationMethod.NEED_BASED
    ) -> Dict[str, float]:
        """Allocate resources among agents using specified method"""
        if not agents:
            return {}
        
        allocation = {}
        
        if method == AllocationMethod.EQUAL:
            per_agent = total_quantity / len(agents)
            allocation = {agent: per_agent for agent in agents}
        
        elif method == AllocationMethod.NEED_BASED:
            # Calculate need scores
            need_scores = {}
            total_need_score = 0.0
            
            for agent in agents:
                agent_needs = [need for need in self.agent_needs[agent] 
                             if need.resource_type == resource_type]
                if agent_needs:
                    # Use highest priority need
                    need = max(agent_needs, key=lambda n: n.priority * n.urgency)
                    need_scores[agent] = need.priority * need.urgency * need.quantity_needed
                else:
                    need_scores[agent] = 0.1  # Minimal allocation for no explicit need
                
                total_need_score += need_scores[agent]
            
            # Allocate proportionally to need
            for agent in agents:
                if total_need_score > 0:
                    allocation[agent] = (need_scores[agent] / total_need_score) * total_quantity
                else:
                    allocation[agent] = total_quantity / len(agents)
        
        elif method == AllocationMethod.LOTTERY:
            # Random allocation
            weights = [random.random() for _ in agents]
            total_weight = sum(weights)
            
            for i, agent in enumerate(agents):
                allocation[agent] = (weights[i] / total_weight) * total_quantity
        
        # Ensure non-negative allocations
        for agent in allocation:
            allocation[agent] = max(0.0, allocation[agent])
        
        return allocation
    
    def get_economic_metrics(self) -> Dict[str, Any]:
        """Get comprehensive economic metrics"""
        metrics = {
            'total_agents': len(self.agents),
            'total_resources': len(self.resources),
            'total_currency': sum(self.currency_balances.values()),
            'active_trades': len([o for o in self.trade_offers.values() if o.status == "pending"]),
            'resource_diversity': len(set(r.resource_type for r in self.resources.values())),
            'average_wealth': mean(self.currency_balances.values()) if self.currency_balances else 0.0,
            'wealth_inequality': self._calculate_wealth_inequality(),
            'resource_concentration': self._calculate_resource_concentration(),
            'market_activity': self._calculate_market_activity(),
            'scarcity_levels': {rt.value: self._get_scarcity_level(rt) for rt in ResourceType},
            'price_levels': {rt.value: self.get_current_price(rt) for rt in ResourceType},
            'trade_efficiency': self._calculate_trade_efficiency(),
            'economic_growth': self._calculate_economic_growth()
        }
        
        return metrics
    
    def export_economic_data(self) -> Dict[str, Any]:
        """Export all economic data for analysis"""
        export_data = {
            'agents': list(self.agents),
            'resources': {rid: asdict(resource) for rid, resource in self.resources.items()},
            'inventories': {aid: dict(inv) for aid, inv in self.agent_inventory.items()},
            'currency_balances': dict(self.currency_balances),
            'trade_offers': {oid: asdict(offer) for oid, offer in self.trade_offers.items()},
            'markets': {mid: asdict(market) for mid, market in self.markets.items()},
            'needs': {aid: [asdict(need) for need in needs] for aid, needs in self.agent_needs.items()},
            'production_rates': {aid: dict(rates) for aid, rates in self.production_rates.items()},
            'consumption_rates': {aid: dict(rates) for aid, rates in self.consumption_rates.items()},
            'transaction_history': self.transaction_history[-1000:],  # Last 1000 transactions
            'economic_metrics': self.get_economic_metrics(),
            'system_configuration': {
                'base_prices': {rt.value: price for rt, price in self.base_prices.items()},
                'scarcity_thresholds': {rt.value: thresh for rt, thresh in self.scarcity_thresholds.items()},
                'inflation_rate': self.inflation_rate,
                'interest_rate': self.interest_rate,
                'transaction_fee_rate': self.transaction_fee_rate
            }
        }
        
        return export_data
    
    # Private helper methods
    
    def _initialize_resource_config(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Initialize resource type configurations"""
        return {
            ResourceType.FOOD: {
                "category": ResourceCategory.CONSUMABLE,
                "shelf_life": 86400 * 7,  # 1 week
                "maintenance_cost": 0.01,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.WATER: {
                "category": ResourceCategory.CONSUMABLE,
                "shelf_life": None,
                "maintenance_cost": 0.0,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.SHELTER: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.1,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.ENERGY: {
                "category": ResourceCategory.CONSUMABLE,
                "maintenance_cost": 0.0,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.WOOD: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.02,
                "transferable": True,
                "initial_durability": 0.9
            },
            ResourceType.STONE: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.001,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.METAL: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.05,
                "transferable": True,
                "initial_durability": 0.95
            },
            ResourceType.FABRIC: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.03,
                "transferable": True,
                "initial_durability": 0.8
            },
            ResourceType.TOOLS: {
                "category": ResourceCategory.DURABLE,
                "maintenance_cost": 0.1,
                "transferable": True,
                "initial_durability": 0.9
            },
            ResourceType.TECHNOLOGY: {
                "category": ResourceCategory.ABSTRACT,
                "maintenance_cost": 0.2,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.INFORMATION: {
                "category": ResourceCategory.ABSTRACT,
                "maintenance_cost": 0.0,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.KNOWLEDGE: {
                "category": ResourceCategory.ABSTRACT,
                "maintenance_cost": 0.05,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.REPUTATION: {
                "category": ResourceCategory.SOCIAL,
                "maintenance_cost": 0.1,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.INFLUENCE: {
                "category": ResourceCategory.SOCIAL,
                "maintenance_cost": 0.15,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.TRUST: {
                "category": ResourceCategory.SOCIAL,
                "maintenance_cost": 0.05,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.SOCIAL_CAPITAL: {
                "category": ResourceCategory.SOCIAL,
                "maintenance_cost": 0.1,
                "transferable": False,
                "initial_durability": 1.0
            },
            ResourceType.CURRENCY: {
                "category": ResourceCategory.ABSTRACT,
                "maintenance_cost": 0.0,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.CREDIT: {
                "category": ResourceCategory.ABSTRACT,
                "maintenance_cost": 0.05,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.LAND: {
                "category": ResourceCategory.FINITE,
                "maintenance_cost": 0.02,
                "transferable": True,
                "initial_durability": 1.0
            },
            ResourceType.LABOR: {
                "category": ResourceCategory.RENEWABLE,
                "maintenance_cost": 0.0,
                "transferable": True,
                "initial_durability": 1.0
            }
        }
    
    def _initialize_base_prices(self) -> Dict[ResourceType, float]:
        """Initialize base prices for resource types"""
        return {
            ResourceType.FOOD: 2.0,
            ResourceType.WATER: 1.0,
            ResourceType.SHELTER: 50.0,
            ResourceType.ENERGY: 3.0,
            ResourceType.WOOD: 5.0,
            ResourceType.STONE: 3.0,
            ResourceType.METAL: 10.0,
            ResourceType.FABRIC: 8.0,
            ResourceType.TOOLS: 20.0,
            ResourceType.TECHNOLOGY: 100.0,
            ResourceType.INFORMATION: 5.0,
            ResourceType.KNOWLEDGE: 25.0,
            ResourceType.REPUTATION: 15.0,
            ResourceType.INFLUENCE: 30.0,
            ResourceType.TRUST: 20.0,
            ResourceType.SOCIAL_CAPITAL: 25.0,
            ResourceType.CURRENCY: 1.0,
            ResourceType.CREDIT: 1.1,
            ResourceType.LAND: 100.0,
            ResourceType.LABOR: 10.0
        }
    
    def _initialize_scarcity_thresholds(self) -> Dict[ResourceType, float]:
        """Initialize scarcity thresholds (total quantity below which resource is considered scarce)"""
        return {
            ResourceType.FOOD: 100.0,
            ResourceType.WATER: 500.0,
            ResourceType.SHELTER: 10.0,
            ResourceType.ENERGY: 200.0,
            ResourceType.WOOD: 100.0,
            ResourceType.STONE: 200.0,
            ResourceType.METAL: 50.0,
            ResourceType.FABRIC: 50.0,
            ResourceType.TOOLS: 20.0,
            ResourceType.TECHNOLOGY: 5.0,
            ResourceType.INFORMATION: 1000.0,
            ResourceType.KNOWLEDGE: 100.0,
            ResourceType.REPUTATION: 50.0,
            ResourceType.INFLUENCE: 25.0,
            ResourceType.TRUST: 100.0,
            ResourceType.SOCIAL_CAPITAL: 50.0,
            ResourceType.CURRENCY: 1000.0,
            ResourceType.CREDIT: 500.0,
            ResourceType.LAND: 20.0,
            ResourceType.LABOR: 100.0
        }
    
    def _initialize_agent_needs(self, agent_id: str):
        """Initialize default needs for a new agent"""
        # Basic survival needs
        self.add_need(agent_id, ResourceType.FOOD, 5.0, priority=0.9, urgency=0.8)
        self.add_need(agent_id, ResourceType.WATER, 10.0, priority=1.0, urgency=0.9)
        self.add_need(agent_id, ResourceType.SHELTER, 1.0, priority=0.7, urgency=0.6)
        self.add_need(agent_id, ResourceType.ENERGY, 3.0, priority=0.6, urgency=0.5)
    
    def _calculate_supply_demand(self, resource_type: ResourceType) -> Tuple[float, float]:
        """Calculate current supply and demand for a resource type"""
        if self._is_cache_valid() and resource_type in self.supply_demand_cache:
            return self.supply_demand_cache[resource_type]
        
        # Calculate supply (total available quantity)
        supply = 0.0
        for resource in self.resources.values():
            if resource.resource_type == resource_type:
                supply += resource.quantity
        
        # Calculate demand (total needed quantity)
        demand = 0.0
        for agent_needs in self.agent_needs.values():
            for need in agent_needs:
                if need.resource_type == resource_type:
                    # Weight by priority and urgency
                    weighted_demand = need.quantity_needed * need.priority * need.urgency
                    demand += weighted_demand
        
        # Add base demand (minimum expected demand)
        base_demand = len(self.agents) * 1.0  # Each agent needs some of each resource
        demand = max(demand, base_demand)
        
        self.supply_demand_cache[resource_type] = (supply, demand)
        return supply, demand
    
    def _calculate_scarcity_level(self, resource_type: ResourceType, total_quantity: float) -> float:
        """Calculate scarcity level (0.0 = abundant, 1.0 = extremely scarce)"""
        threshold = self.scarcity_thresholds[resource_type]
        if total_quantity >= threshold:
            return 0.0
        return 1.0 - (total_quantity / threshold)
    
    def _get_scarcity_level(self, resource_type: ResourceType) -> float:
        """Get current scarcity level for a resource type"""
        availability = self.get_resource_availability(resource_type)
        return availability['scarcity_level']
    
    def _execute_trade(self, offer: TradeOffer, accepter_id: str) -> bool:
        """Execute a trade between proposer and accepter"""
        try:
            # Transfer offered resources to accepter
            for resource_id, quantity in offer.offered_resources.items():
                success = self.transfer_resource(resource_id, offer.proposer_id, accepter_id, quantity)
                if not success:
                    return False
            
            # Transfer currency from proposer to accepter
            if offer.currency_offered > 0:
                if self.currency_balances[offer.proposer_id] >= offer.currency_offered:
                    self.currency_balances[offer.proposer_id] -= offer.currency_offered
                    self.currency_balances[accepter_id] += offer.currency_offered
                else:
                    return False
            
            # Transfer requested resources to proposer
            for resource_type, quantity in offer.requested_resources.items():
                consumed = self.consume_resource(accepter_id, resource_type, quantity)
                if consumed < quantity * 0.9:  # Allow 10% shortage
                    return False
                
                # Create new resource for proposer
                self.create_resource(resource_type, consumed, offer.proposer_id)
            
            # Transfer currency from accepter to proposer
            if offer.currency_requested > 0:
                if self.currency_balances[accepter_id] >= offer.currency_requested:
                    self.currency_balances[accepter_id] -= offer.currency_requested
                    self.currency_balances[offer.proposer_id] += offer.currency_requested
                else:
                    return False
            
            # Apply transaction fees
            total_value = offer.currency_offered + offer.currency_requested
            fee = round(total_value * self.transaction_fee_rate, 2)
            
            self.currency_balances[offer.proposer_id] = round(self.currency_balances[offer.proposer_id] - fee / 2, 2)
            self.currency_balances[accepter_id] = round(self.currency_balances[accepter_id] - fee / 2, 2)
            
            return True
            
        except Exception as e:
            return False
    
    def _record_transaction(
        self, 
        from_agent: str, 
        to_agent: str, 
        resource_id: str, 
        quantity: float, 
        value: float, 
        transaction_type: str
    ):
        """Record a transaction in the history"""
        transaction = {
            'timestamp': time.time(),
            'from_agent': from_agent,
            'to_agent': to_agent,
            'resource_id': resource_id,
            'quantity': quantity,
            'value': value,
            'type': transaction_type
        }
        
        self.transaction_history.append(transaction)
        
        # Keep only recent transactions
        if len(self.transaction_history) > 10000:
            self.transaction_history = self.transaction_history[-5000:]
    
    def _calculate_wealth_inequality(self) -> float:
        """Calculate wealth inequality using Gini coefficient"""
        if len(self.currency_balances) < 2:
            return 0.0
        
        wealth_values = list(self.currency_balances.values())
        wealth_values.sort()
        n = len(wealth_values)
        
        if sum(wealth_values) == 0:
            return 0.0
        
        # Calculate Gini coefficient
        total = sum(wealth_values)
        gini_sum = sum((2 * i - n - 1) * wealth for i, wealth in enumerate(wealth_values, 1))
        gini = gini_sum / (n * total)
        
        return gini
    
    def _calculate_resource_concentration(self) -> float:
        """Calculate how concentrated resource ownership is"""
        if not self.resources:
            return 0.0
        
        owner_counts = defaultdict(int)
        for resource in self.resources.values():
            if resource.owner_id:
                owner_counts[resource.owner_id] += 1
        
        if not owner_counts:
            return 0.0
        
        total_resources = len(self.resources)
        max_owned = max(owner_counts.values())
        
        return max_owned / total_resources
    
    def _calculate_market_activity(self) -> float:
        """Calculate market activity level"""
        recent_transactions = [t for t in self.transaction_history 
                             if time.time() - t['timestamp'] < 86400]  # Last 24 hours
        
        if not recent_transactions:
            return 0.0
        
        # Activity based on transaction volume and frequency
        total_value = sum(t['value'] for t in recent_transactions)
        transaction_count = len(recent_transactions)
        
        # Normalize to agent count
        activity_score = (total_value / max(len(self.agents), 1)) + (transaction_count / max(len(self.agents), 1))
        
        return min(1.0, activity_score / 100.0)  # Normalize to 0-1
    
    def _calculate_trade_efficiency(self) -> float:
        """Calculate trade efficiency based on successful vs failed trades"""
        recent_offers = [o for o in self.trade_offers.values() 
                        if time.time() - o.created_time < 604800]  # Last week
        
        if not recent_offers:
            return 0.5  # Neutral
        
        completed_offers = [o for o in recent_offers if o.status == "completed"]
        return len(completed_offers) / len(recent_offers)
    
    def _calculate_economic_growth(self) -> float:
        """Calculate economic growth rate"""
        # Simple measure: change in total wealth over time
        current_total_wealth = sum(self.currency_balances.values())
        
        # This would need historical data for proper calculation
        # For now, return a placeholder based on transaction activity
        recent_activity = self._calculate_market_activity()
        return recent_activity * 0.1  # Convert activity to growth rate
    
    def _invalidate_cache(self):
        """Invalidate cached values"""
        self.supply_demand_cache.clear()
        self.price_cache.clear()
        self._cache_timestamp = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return (time.time() - self._cache_timestamp) < self._cache_ttl