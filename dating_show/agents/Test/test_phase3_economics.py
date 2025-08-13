"""
Comprehensive Test Suite for Phase 3 Economic Systems
Testing: Resource Management, Economic Metrics, Trade Systems

This test suite validates the functionality, performance, and integration
of the economic systems implemented in Phase 3.
"""

import unittest
import time
import json
import threading
from unittest.mock import patch, MagicMock
from typing import Dict, List, Set, Any

# Import the Phase 3 economics modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../economics'))

from resources import (
    ResourceManagementSystem, ResourceType, ResourceCategory, AllocationMethod,
    ResourceInstance, ResourceNeed, TradeOffer, Market
)


class TestResourceManagement(unittest.TestCase):
    """Test suite for ResourceManagementSystem"""
    
    def setUp(self):
        """Set up test environment"""
        self.resource_system = ResourceManagementSystem(max_agents=100, max_resources=1000)
        self.test_agents = ["alice", "bob", "charlie", "diana", "eve"]
        
        # Add test agents with initial resources
        for i, agent in enumerate(self.test_agents):
            initial_resources = {
                ResourceType.FOOD: 10.0 + i * 5,
                ResourceType.WATER: 20.0 + i * 3,
                ResourceType.CURRENCY: 100.0 + i * 50
            }
            self.resource_system.add_agent(agent, 100.0 + i * 50, initial_resources)
    
    def tearDown(self):
        """Clean up test environment"""
        self.resource_system = None
    
    def test_agent_management(self):
        """Test agent addition and removal"""
        # Verify agents added
        self.assertEqual(len(self.resource_system.agents), 5)
        
        # Verify initial currency
        for i, agent in enumerate(self.test_agents):
            expected_currency = 100.0 + i * 50
            self.assertEqual(self.resource_system.currency_balances[agent], expected_currency)
        
        # Verify initial resources
        alice_inventory = self.resource_system.get_agent_inventory("alice")
        self.assertIn(ResourceType.FOOD, alice_inventory)
        self.assertGreater(alice_inventory[ResourceType.FOOD], 0)
        
        # Test removal
        result = self.resource_system.remove_agent("eve")
        self.assertTrue(result)
        self.assertEqual(len(self.resource_system.agents), 4)
        self.assertNotIn("eve", self.resource_system.currency_balances)
    
    def test_resource_creation(self):
        """Test resource creation and management"""
        # Create various resource types
        food_id = self.resource_system.create_resource(
            ResourceType.FOOD, 5.0, "alice", quality=0.8
        )
        self.assertNotEqual(food_id, "")
        self.assertIn(food_id, self.resource_system.resources)
        
        # Verify resource properties
        food_resource = self.resource_system.resources[food_id]
        self.assertEqual(food_resource.resource_type, ResourceType.FOOD)
        self.assertEqual(food_resource.quantity, 5.0)
        self.assertEqual(food_resource.quality, 0.8)
        self.assertEqual(food_resource.owner_id, "alice")
        
        # Create durable resource
        tool_id = self.resource_system.create_resource(
            ResourceType.TOOLS, 1.0, "bob", quality=0.9
        )
        tool_resource = self.resource_system.resources[tool_id]
        self.assertEqual(tool_resource.category, ResourceCategory.DURABLE)
        
        # Create abstract resource
        info_id = self.resource_system.create_resource(
            ResourceType.INFORMATION, 3.0, "charlie"
        )
        info_resource = self.resource_system.resources[info_id]
        self.assertEqual(info_resource.category, ResourceCategory.ABSTRACT)
    
    def test_resource_transfer(self):
        """Test resource transfers between agents"""
        # Create resource for Alice
        food_id = self.resource_system.create_resource(
            ResourceType.FOOD, 10.0, "alice"
        )
        
        # Transfer to Bob
        result = self.resource_system.transfer_resource(food_id, "alice", "bob", 5.0)
        self.assertTrue(result)
        
        # Verify transfer
        alice_inventory = self.resource_system.get_agent_inventory("alice")
        bob_inventory = self.resource_system.get_agent_inventory("bob")
        
        self.assertEqual(alice_inventory[ResourceType.FOOD], 15.0)  # 10 initial + 10 created - 5 transferred
        self.assertGreater(bob_inventory[ResourceType.FOOD], 15.0)  # Initial + transfer
        
        # Test full transfer
        result = self.resource_system.transfer_resource(food_id, "alice", "charlie")
        self.assertTrue(result)
        
        # Alice should have no more of this resource
        alice_new_inventory = self.resource_system.get_agent_inventory("alice")
        self.assertLess(alice_new_inventory.get(ResourceType.FOOD, 0), 15.0)
        
        # Ownership should transfer
        resource = self.resource_system.resources[food_id]
        self.assertEqual(resource.owner_id, "charlie")
    
    def test_resource_consumption(self):
        """Test resource consumption"""
        # Alice starts with food, let's add more
        food_id = self.resource_system.create_resource(
            ResourceType.FOOD, 15.0, "alice", quality=0.9
        )
        
        initial_food = self.resource_system.get_agent_inventory("alice")[ResourceType.FOOD]
        
        # Consume food
        consumed = self.resource_system.consume_resource("alice", ResourceType.FOOD, 8.0)
        self.assertEqual(consumed, 8.0)
        
        # Verify consumption
        final_food = self.resource_system.get_agent_inventory("alice")[ResourceType.FOOD]
        self.assertEqual(final_food, initial_food - 8.0)
        
        # Test consumption with efficiency
        consumed_efficient = self.resource_system.consume_resource(
            "alice", ResourceType.FOOD, 5.0, efficiency=2.0
        )
        self.assertEqual(consumed_efficient, 10.0)  # 5.0 * 2.0 efficiency
        
        # Test over-consumption
        large_consumption = self.resource_system.consume_resource(
            "alice", ResourceType.FOOD, 1000.0
        )
        self.assertLess(large_consumption, 1000.0)  # Can't consume more than available
    
    def test_resource_production(self):
        """Test resource production"""
        # Simple production without inputs
        wood_id = self.resource_system.produce_resource(
            "alice", ResourceType.WOOD, 5.0, efficiency=1.2
        )
        self.assertNotEqual(wood_id, "")
        
        # Verify production
        wood_resource = self.resource_system.resources[wood_id]
        self.assertEqual(wood_resource.quantity, 6.0)  # 5.0 * 1.2 efficiency
        self.assertEqual(wood_resource.owner_id, "alice")
        
        # Production with inputs
        self.resource_system.create_resource(ResourceType.WOOD, 10.0, "bob")
        self.resource_system.create_resource(ResourceType.METAL, 5.0, "bob")
        
        tool_id = self.resource_system.produce_resource(
            "bob", ResourceType.TOOLS, 2.0,
            inputs={ResourceType.WOOD: 3.0, ResourceType.METAL: 2.0},
            efficiency=1.0
        )
        self.assertNotEqual(tool_id, "")
        
        # Verify inputs consumed
        bob_inventory = self.resource_system.get_agent_inventory("bob")
        self.assertLess(bob_inventory[ResourceType.WOOD], 10.0)
        self.assertLess(bob_inventory[ResourceType.METAL], 5.0)
    
    def test_resource_availability(self):
        """Test resource availability queries"""
        # Create resources
        self.resource_system.create_resource(ResourceType.FOOD, 15.0, "alice", quality=0.8)
        self.resource_system.create_resource(ResourceType.FOOD, 10.0, "bob", quality=0.9)
        self.resource_system.create_resource(ResourceType.FOOD, 8.0, None, quality=0.7)  # Ownerless
        
        # Get availability
        availability = self.resource_system.get_resource_availability(ResourceType.FOOD)
        
        # Verify availability data
        self.assertGreater(availability["total_quantity"], 33.0)  # Initial + created
        self.assertGreater(availability["total_value"], 0.0)
        self.assertGreater(availability["average_quality"], 0.7)
        self.assertGreaterEqual(availability["num_owners"], 2)
        self.assertIn("scarcity_level", availability)
        
        # Test location filtering
        self.resource_system.create_resource(ResourceType.WATER, 20.0, "alice", location="location_a")
        location_availability = self.resource_system.get_resource_availability(
            ResourceType.WATER, "location_a"
        )
        self.assertGreater(location_availability["total_quantity"], 0.0)
    
    def test_trade_offers(self):
        """Test trade offer creation and acceptance"""
        # Create resources for trading
        food_id = self.resource_system.create_resource(ResourceType.FOOD, 10.0, "alice")
        water_id = self.resource_system.create_resource(ResourceType.WATER, 15.0, "bob")
        
        # Alice offers food for currency
        offer_id = self.resource_system.create_trade_offer(
            "alice", {food_id: 5.0}, {}, currency_offered=0.0, currency_requested=20.0
        )
        self.assertNotEqual(offer_id, "")
        
        # Verify offer
        offer = self.resource_system.trade_offers[offer_id]
        self.assertEqual(offer.proposer_id, "alice")
        self.assertEqual(offer.currency_requested, 20.0)
        self.assertIn(food_id, offer.offered_resources)
        
        # Bob accepts the offer
        result = self.resource_system.accept_trade_offer(offer_id, "bob")
        self.assertTrue(result)
        
        # Verify trade executed
        offer_after = self.resource_system.trade_offers[offer_id]
        self.assertEqual(offer_after.status, "completed")
        
        # Verify balances
        self.assertEqual(self.resource_system.currency_balances["alice"], 119.9)  # 100 + 20 - 0.1 fee
        self.assertEqual(self.resource_system.currency_balances["bob"], 129.9)   # 150 - 20 - 0.1 fee
        
        # Verify resource transfer
        bob_inventory = self.resource_system.get_agent_inventory("bob")
        self.assertGreater(bob_inventory[ResourceType.FOOD], 5.0)
    
    def test_resource_allocation(self):
        """Test resource allocation algorithms"""
        # Create resources to allocate
        total_food = 30.0
        agents = ["alice", "bob", "charlie"]
        
        # Test equal allocation
        equal_allocation = self.resource_system.allocate_resources(
            ResourceType.FOOD, total_food, agents, AllocationMethod.EQUAL
        )
        
        for agent in agents:
            self.assertEqual(equal_allocation[agent], 10.0)
        
        # Test need-based allocation
        # Add needs for agents
        self.resource_system.add_need("alice", ResourceType.FOOD, 15.0, priority=0.9, urgency=0.8)
        self.resource_system.add_need("bob", ResourceType.FOOD, 8.0, priority=0.6, urgency=0.5)
        self.resource_system.add_need("charlie", ResourceType.FOOD, 5.0, priority=0.4, urgency=0.3)
        
        need_allocation = self.resource_system.allocate_resources(
            ResourceType.FOOD, total_food, agents, AllocationMethod.NEED_BASED
        )
        
        # Alice should get more due to higher priority and urgency
        # Alice: 0.9*0.8*15.0=10.8, Bob: 0.6*0.5*8.0=2.4, Charlie: 0.4*0.3*5.0=0.6
        # Total score: 13.8, Alice should get: (10.8/13.8)*30≈23.5
        self.assertGreater(need_allocation["alice"], need_allocation["bob"])
        self.assertGreater(need_allocation["alice"], 20.0)  # Should get significant majority
        self.assertGreater(need_allocation["bob"], need_allocation["charlie"])
        
        # Test lottery allocation
        lottery_allocation = self.resource_system.allocate_resources(
            ResourceType.FOOD, total_food, agents, AllocationMethod.LOTTERY
        )
        
        # Should allocate all resources
        total_allocated = sum(lottery_allocation.values())
        self.assertAlmostEqual(total_allocated, total_food, places=2)
    
    def test_pricing_dynamics(self):
        """Test dynamic pricing based on supply and demand"""
        # Get initial price
        initial_price = self.resource_system.get_current_price(ResourceType.FOOD)
        self.assertGreater(initial_price, 0.0)
        
        # Create high demand scenario
        for agent in self.test_agents:
            self.resource_system.add_need(agent, ResourceType.FOOD, 50.0, priority=0.9, urgency=0.9)
        
        # Price should increase due to high demand
        high_demand_price = self.resource_system.get_current_price(ResourceType.FOOD)
        self.assertGreater(high_demand_price, initial_price)
        
        # Create high supply scenario
        for i in range(10):
            self.resource_system.create_resource(ResourceType.FOOD, 20.0, "alice")
        
        # Price should decrease due to high supply
        high_supply_price = self.resource_system.get_current_price(ResourceType.FOOD)
        self.assertLess(high_supply_price, high_demand_price)
    
    def test_market_creation(self):
        """Test marketplace creation and management"""
        # Create market
        result = self.resource_system.create_market(
            "central_market", "Central Marketplace", "town_center",
            {ResourceType.FOOD, ResourceType.WATER, ResourceType.TOOLS},
            transaction_fee=0.02, reputation_required=0.0
        )
        self.assertTrue(result)
        
        # Verify market created
        self.assertIn("central_market", self.resource_system.markets)
        market = self.resource_system.markets["central_market"]
        
        self.assertEqual(market.name, "Central Marketplace")
        self.assertEqual(market.location, "town_center")
        self.assertEqual(market.transaction_fee, 0.02)
        self.assertIn(ResourceType.FOOD, market.resource_types)
    
    def test_economic_metrics(self):
        """Test economic metrics calculation"""
        # Set up diverse economic scenario
        self.resource_system.currency_balances["alice"] = 500.0
        self.resource_system.currency_balances["bob"] = 100.0
        self.resource_system.currency_balances["charlie"] = 300.0
        
        # Create and execute some trades
        food_id = self.resource_system.create_resource(ResourceType.FOOD, 10.0, "alice")
        offer_id = self.resource_system.create_trade_offer(
            "alice", {food_id: 5.0}, {}, currency_requested=25.0
        )
        self.resource_system.accept_trade_offer(offer_id, "bob")
        
        # Get metrics
        metrics = self.resource_system.get_economic_metrics()
        
        # Verify metrics structure
        expected_metrics = [
            "total_agents", "total_resources", "total_currency", "active_trades",
            "resource_diversity", "average_wealth", "wealth_inequality",
            "resource_concentration", "market_activity", "scarcity_levels",
            "price_levels", "trade_efficiency", "economic_growth"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Verify values
        self.assertEqual(metrics["total_agents"], len(self.test_agents))
        self.assertGreater(metrics["total_currency"], 0.0)
        self.assertGreaterEqual(metrics["wealth_inequality"], 0.0)
        self.assertLessEqual(metrics["wealth_inequality"], 1.0)
    
    def test_scarcity_management(self):
        """Test scarcity detection and management"""
        # Create scarcity scenario
        # Consume most of the food
        for agent in self.test_agents[:3]:
            inventory = self.resource_system.get_agent_inventory(agent)
            if ResourceType.FOOD in inventory:
                self.resource_system.consume_resource(
                    agent, ResourceType.FOOD, inventory[ResourceType.FOOD] * 0.9
                )
        
        # Check scarcity level
        availability = self.resource_system.get_resource_availability(ResourceType.FOOD)
        scarcity_level = availability["scarcity_level"]
        
        if availability["total_quantity"] < self.resource_system.scarcity_thresholds[ResourceType.FOOD]:
            self.assertGreater(scarcity_level, 0.0)
        
        # Price should be affected by scarcity
        scarce_price = self.resource_system.get_current_price(ResourceType.FOOD)
        base_price = self.resource_system.base_prices[ResourceType.FOOD]
        
        if scarcity_level > 0:
            self.assertGreater(scarce_price, base_price)
    
    def test_needs_management(self):
        """Test agent needs tracking and management"""
        # Add various needs
        result = self.resource_system.add_need(
            "alice", ResourceType.SHELTER, 1.0, priority=0.8, urgency=0.7
        )
        self.assertTrue(result)
        
        result = self.resource_system.add_need(
            "alice", ResourceType.TOOLS, 2.0, priority=0.6, urgency=0.4, max_price=50.0
        )
        self.assertTrue(result)
        
        # Verify needs added
        alice_needs = self.resource_system.agent_needs["alice"]
        shelter_needs = [n for n in alice_needs if n.resource_type == ResourceType.SHELTER]
        tool_needs = [n for n in alice_needs if n.resource_type == ResourceType.TOOLS]
        
        self.assertGreaterEqual(len(shelter_needs), 1)  # May have multiple shelter needs from other tests
        self.assertEqual(len(tool_needs), 1)
        
        # Verify need properties
        shelter_need = shelter_needs[0]
        self.assertEqual(shelter_need.quantity_needed, 1.0)
        self.assertEqual(shelter_need.priority, 0.8)
        self.assertEqual(shelter_need.urgency, 0.7)
        
        tool_need = tool_needs[0]
        self.assertEqual(tool_need.max_price, 50.0)
    
    def test_performance(self):
        """Test resource system performance"""
        # Create large system
        large_system = ResourceManagementSystem(max_agents=500, max_resources=5000)
        
        # Add many agents
        start_time = time.time()
        for i in range(200):
            large_system.add_agent(f"agent_{i}", 100.0, {ResourceType.CURRENCY: 100.0})
        agent_creation_time = time.time() - start_time
        
        self.assertLess(agent_creation_time, 3.0)
        
        # Create many resources
        start_time = time.time()
        for i in range(500):
            large_system.create_resource(
                ResourceType.FOOD, 10.0, f"agent_{i % 200}"
            )
        resource_creation_time = time.time() - start_time
        
        self.assertLess(resource_creation_time, 5.0)
        
        # Test pricing calculation performance
        start_time = time.time()
        price = large_system.get_current_price(ResourceType.FOOD)
        pricing_time = time.time() - start_time
        
        self.assertLess(pricing_time, 0.1)
        
        # Test metrics calculation performance
        start_time = time.time()
        metrics = large_system.get_economic_metrics()
        metrics_time = time.time() - start_time
        
        self.assertLess(metrics_time, 2.0)
    
    def test_thread_safety(self):
        """Test thread safety of resource operations"""
        def create_and_trade(agent_prefix, count):
            for i in range(count):
                agent_id = f"{agent_prefix}_{i}"
                self.resource_system.add_agent(agent_id, 100.0)
                
                # Create resource
                resource_id = self.resource_system.create_resource(
                    ResourceType.FOOD, 10.0, agent_id
                )
                
                # Try to trade if resource created
                if resource_id:
                    offer_id = self.resource_system.create_trade_offer(
                        agent_id, {resource_id: 5.0}, {}, currency_requested=10.0
                    )
        
        # Run concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_and_trade, args=(f"thread{i}", 5))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no corruption
        self.assertGreater(len(self.resource_system.agents), 5)
        self.assertGreater(len(self.resource_system.resources), 0)
    
    def test_data_export_import(self):
        """Test economic data export capabilities"""
        # Set up economic scenario
        food_id = self.resource_system.create_resource(ResourceType.FOOD, 15.0, "alice")
        self.resource_system.add_need("bob", ResourceType.FOOD, 10.0, priority=0.8)
        
        offer_id = self.resource_system.create_trade_offer(
            "alice", {food_id: 5.0}, {}, currency_requested=15.0
        )
        
        # Export data
        export_data = self.resource_system.export_economic_data()
        
        # Verify export structure
        expected_sections = [
            "agents", "resources", "inventories", "currency_balances",
            "trade_offers", "needs", "economic_metrics", "system_configuration"
        ]
        
        for section in expected_sections:
            self.assertIn(section, export_data)
        
        # Verify data integrity
        self.assertEqual(len(export_data["agents"]), len(self.test_agents))
        self.assertGreater(len(export_data["resources"]), 0)
        self.assertIn("alice", export_data["currency_balances"])
        
        # Verify configuration data
        config = export_data["system_configuration"]
        self.assertIn("base_prices", config)
        self.assertIn("inflation_rate", config)


class TestEconomicIntegration(unittest.TestCase):
    """Integration tests for economic systems"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.resource_system = ResourceManagementSystem(max_agents=50, max_resources=500)
        self.test_agents = ["alice", "bob", "charlie", "diana"]
        
        # Create diverse economic agents
        agent_profiles = [
            ("alice", {ResourceType.FOOD: 20.0, ResourceType.TOOLS: 2.0}, 200.0),
            ("bob", {ResourceType.WATER: 30.0, ResourceType.WOOD: 15.0}, 150.0),
            ("charlie", {ResourceType.METAL: 10.0, ResourceType.ENERGY: 25.0}, 300.0),
            ("diana", {ResourceType.FABRIC: 12.0, ResourceType.CURRENCY: 50.0}, 100.0)
        ]
        
        for agent, resources, currency in agent_profiles:
            self.resource_system.add_agent(agent, currency, resources)
    
    def test_complete_trade_cycle(self):
        """Test complete trade cycle from need to fulfillment"""
        # 1. Alice needs tools, has food
        self.resource_system.add_need("alice", ResourceType.TOOLS, 1.0, priority=0.9, urgency=0.8)
        
        # 2. Bob has tools, needs food
        tools_id = self.resource_system.create_resource(ResourceType.TOOLS, 2.0, "bob")
        self.resource_system.add_need("bob", ResourceType.FOOD, 8.0, priority=0.7, urgency=0.6)
        
        # 3. Alice creates trade offer
        alice_food = list(self.resource_system.agent_inventory["alice"].keys())[0]
        offer_id = self.resource_system.create_trade_offer(
            "alice", {alice_food: 8.0}, {ResourceType.TOOLS: 1.0}
        )
        
        # 4. Bob accepts trade
        result = self.resource_system.accept_trade_offer(offer_id, "bob")
        self.assertTrue(result)
        
        # 5. Verify trade fulfillment
        alice_inventory = self.resource_system.get_agent_inventory("alice")
        bob_inventory = self.resource_system.get_agent_inventory("bob")
        
        self.assertGreater(alice_inventory.get(ResourceType.TOOLS, 0), 0)
        self.assertGreater(bob_inventory.get(ResourceType.FOOD, 0), 0)
    
    def test_market_dynamics(self):
        """Test market dynamics with multiple agents"""
        # Create central market
        self.resource_system.create_market(
            "central", "Central Market", "center",
            {ResourceType.FOOD, ResourceType.WATER, ResourceType.TOOLS}
        )
        
        # Create supply and demand imbalance
        # High demand for food
        for agent in self.test_agents:
            self.resource_system.add_need(agent, ResourceType.FOOD, 20.0, priority=0.8, urgency=0.7)
        
        # Limited food supply
        for agent in ["alice", "bob"]:
            inventory = self.resource_system.get_agent_inventory(agent)
            if ResourceType.FOOD in inventory:
                # Consume most food to create scarcity
                self.resource_system.consume_resource(
                    agent, ResourceType.FOOD, inventory[ResourceType.FOOD] * 0.8
                )
        
        # Check price impact
        food_price = self.resource_system.get_current_price(ResourceType.FOOD)
        base_price = self.resource_system.base_prices[ResourceType.FOOD]
        
        # Price should be higher due to scarcity and high demand
        self.assertGreaterEqual(food_price, base_price)
        
        # Create trade offers at market prices
        remaining_food = self.resource_system.get_agent_inventory("alice").get(ResourceType.FOOD, 0)
        if remaining_food > 0:
            alice_food_resources = [
                rid for rid, resource in self.resource_system.resources.items()
                if resource.resource_type == ResourceType.FOOD and resource.owner_id == "alice"
            ]
            
            if alice_food_resources:
                offer_id = self.resource_system.create_trade_offer(
                    "alice", {alice_food_resources[0]: min(5.0, remaining_food)}, {},
                    currency_requested=food_price * 5.0
                )
                
                # Charlie (with most currency) should be able to accept
                result = self.resource_system.accept_trade_offer(offer_id, "charlie")
                if self.resource_system.currency_balances["charlie"] >= food_price * 5.0:
                    self.assertTrue(result)
    
    def test_production_chain(self):
        """Test complex production chains"""
        # Create production chain: Wood + Metal -> Tools -> Complex Tools
        
        # 1. Charlie produces basic tools from raw materials
        charlie_metal = self.resource_system.get_agent_inventory("charlie").get(ResourceType.METAL, 0)
        
        if charlie_metal >= 2.0:
            # Add wood to charlie
            wood_id = self.resource_system.create_resource(ResourceType.WOOD, 5.0, "charlie")
            
            # Produce tools
            tools_id = self.resource_system.produce_resource(
                "charlie", ResourceType.TOOLS, 1.0,
                inputs={ResourceType.WOOD: 3.0, ResourceType.METAL: 2.0},
                efficiency=1.0
            )
            self.assertNotEqual(tools_id, "")
        
        # 2. Diana produces technology from tools and energy
        diana_energy = self.resource_system.get_agent_inventory("diana").get(ResourceType.ENERGY, 0)
        
        # Transfer tools to diana for tech production
        if tools_id:
            self.resource_system.transfer_resource(tools_id, "charlie", "diana", 1.0)
            
            # Add energy to diana if needed
            if diana_energy < 5.0:
                energy_id = self.resource_system.create_resource(ResourceType.ENERGY, 10.0, "diana")
            
            # Produce technology
            tech_id = self.resource_system.produce_resource(
                "diana", ResourceType.TECHNOLOGY, 1.0,
                inputs={ResourceType.TOOLS: 1.0, ResourceType.ENERGY: 5.0},
                efficiency=0.8
            )
            
            if diana_energy >= 5.0:
                self.assertNotEqual(tech_id, "")
    
    def test_economic_inequality(self):
        """Test economic inequality measurement and effects"""
        # Create wealth disparity
        self.resource_system.currency_balances["alice"] = 1000.0
        self.resource_system.currency_balances["bob"] = 50.0
        self.resource_system.currency_balances["charlie"] = 500.0
        self.resource_system.currency_balances["diana"] = 25.0
        
        # Calculate inequality
        metrics = self.resource_system.get_economic_metrics()
        inequality = metrics["wealth_inequality"]
        
        # Should show significant inequality
        self.assertGreater(inequality, 0.3)  # Gini coefficient > 0.3 indicates inequality
        
        # Test inequality effects on trade
        # Poor agents should have limited trade options
        expensive_resource_id = self.resource_system.create_resource(
            ResourceType.TECHNOLOGY, 1.0, "alice"
        )
        
        expensive_offer = self.resource_system.create_trade_offer(
            "alice", {expensive_resource_id: 1.0}, {},
            currency_requested=500.0
        )
        
        # Diana (poorest) cannot afford this
        result = self.resource_system.accept_trade_offer(expensive_offer, "diana")
        self.assertFalse(result)
        
        # Charlie also cannot afford this (only has 200, needs 500)
        result = self.resource_system.accept_trade_offer(expensive_offer, "charlie")
        self.assertFalse(result)
    
    def test_resource_allocation_fairness(self):
        """Test fair resource allocation under scarcity"""
        # Create scarcity scenario
        scarce_resource_total = 20.0
        
        # Add high needs for everyone
        for agent in self.test_agents:
            self.resource_system.add_need(
                agent, ResourceType.SHELTER, 8.0, priority=0.9, urgency=0.8
            )
        
        # Test different allocation methods
        allocations = {}
        
        # Equal allocation
        allocations["equal"] = self.resource_system.allocate_resources(
            ResourceType.SHELTER, scarce_resource_total, self.test_agents, AllocationMethod.EQUAL
        )
        
        # Need-based allocation (should be similar since all have same needs)
        allocations["need_based"] = self.resource_system.allocate_resources(
            ResourceType.SHELTER, scarce_resource_total, self.test_agents, AllocationMethod.NEED_BASED
        )
        
        # Verify allocations
        for method, allocation in allocations.items():
            total_allocated = sum(allocation.values())
            self.assertAlmostEqual(total_allocated, scarce_resource_total, places=2)
            
            # All agents should get something
            for agent in self.test_agents:
                self.assertGreater(allocation[agent], 0.0)
        
        # Equal allocation should be exactly equal
        equal_share = scarce_resource_total / len(self.test_agents)
        for agent in self.test_agents:
            self.assertAlmostEqual(allocations["equal"][agent], equal_share, places=2)
    
    def test_economic_growth(self):
        """Test economic growth through production and trade"""
        # Measure initial economic state
        initial_metrics = self.resource_system.get_economic_metrics()
        initial_wealth = initial_metrics["total_currency"]
        initial_resources = initial_metrics["total_resources"]
        
        # Simulate economic activity
        # 1. Production increases total resources
        for agent in self.test_agents[:2]:
            self.resource_system.produce_resource(agent, ResourceType.FOOD, 5.0)
        
        # 2. Trade redistributes wealth and creates value
        food_resources = [
            (rid, resource) for rid, resource in self.resource_system.resources.items()
            if (resource.resource_type == ResourceType.FOOD and 
                resource.owner_id in self.test_agents[:2])
        ]
        
        if food_resources:
            food_id, food_resource = food_resources[0]
            offer_id = self.resource_system.create_trade_offer(
                food_resource.owner_id, {food_id: 3.0}, {},
                currency_requested=15.0
            )
            
            # Find agent who can afford this
            for agent in self.test_agents:
                if (agent != food_resource.owner_id and 
                    self.resource_system.currency_balances[agent] >= 15.0):
                    self.resource_system.accept_trade_offer(offer_id, agent)
                    break
        
        # 3. Measure growth
        final_metrics = self.resource_system.get_economic_metrics()
        final_resources = final_metrics["total_resources"]
        
        # Should have more resources due to production
        self.assertGreater(final_resources, initial_resources)
        
        # Market activity should be positive
        self.assertGreaterEqual(final_metrics["market_activity"], 0.0)


def run_economics_tests():
    """Run all Phase 3 economics tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestResourceManagement,
        TestEconomicIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "details": {
            "failures": [str(failure) for failure in result.failures],
            "errors": [str(error) for error in result.errors]
        }
    }


if __name__ == "__main__":
    print("Running Phase 3 Economics Test Suite...")
    results = run_economics_tests()
    
    print(f"\nTest Results:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    
    if results['failures'] > 0:
        print(f"\nFailures:")
        for failure in results['details']['failures']:
            print(f"  - {failure}")
    
    if results['errors'] > 0:
        print(f"\nErrors:")
        for error in results['details']['errors']:
            print(f"  - {error}")