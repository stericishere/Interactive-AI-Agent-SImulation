"""
File: constitution_storage.py
Description: Constitutional rule storage and versioning system with LangGraph Store API integration.
Manages constitutional rules, version control, and rule evolution tracking for the collective governance system.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import hashlib

# Import Store integration and governance types
try:
    from ..agents.memory_structures.store_integration import (
        MemoryStoreIntegration, StoreNamespace, GovernanceRule
    )
except ImportError:
    try:
        from dating_show.agents.memory_structures.store_integration import (
            MemoryStoreIntegration, StoreNamespace, GovernanceRule
        )
    except ImportError:
        # Mock for testing
        class MemoryStoreIntegration:
            pass
        class StoreNamespace:
            GOVERNANCE = "governance"
        class GovernanceRule:
            pass


class RuleType(Enum):
    """Types of constitutional rules."""
    FUNDAMENTAL_RIGHT = "fundamental_right"  # Core rights that require supermajority to change
    GOVERNANCE_PROCEDURE = "governance_procedure"  # How decisions are made
    BEHAVIORAL_NORM = "behavioral_norm"  # Expected behaviors and social norms
    RESOURCE_ALLOCATION = "resource_allocation"  # How resources are distributed
    CONFLICT_RESOLUTION = "conflict_resolution"  # How disputes are handled
    AMENDMENT_PROCESS = "amendment_process"  # How rules themselves can be changed


class RuleStatus(Enum):
    """Status of constitutional rules."""
    ACTIVE = "active"  # Currently in effect
    PROPOSED = "proposed"  # Proposed but not yet voted on
    UNDER_REVIEW = "under_review"  # Being considered for changes
    DEPRECATED = "deprecated"  # No longer in effect but kept for history
    SUPERSEDED = "superseded"  # Replaced by a newer version


@dataclass
class ConstitutionalRule:
    """A single constitutional rule with metadata and versioning."""
    rule_id: str
    title: str
    content: str
    rule_type: RuleType
    status: RuleStatus
    version: int
    created_at: datetime
    created_by: str  # Agent ID or system
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    precedence_level: int = 100  # Higher numbers = higher precedence
    requires_supermajority: bool = False
    enforcement_mechanism: str = "community_based"
    related_rules: List[str] = field(default_factory=list)
    violation_penalties: Dict[str, Any] = field(default_factory=dict)
    amendment_history: List[str] = field(default_factory=list)  # Rule IDs of previous versions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.amendment_history:
            self.amendment_history = []
        if not self.related_rules:
            self.related_rules = []
        if not self.violation_penalties:
            self.violation_penalties = {}
        if not self.metadata:
            self.metadata = {}

    def get_rule_hash(self) -> str:
        """Generate a hash for the rule content for integrity checking."""
        content_str = f"{self.title}|{self.content}|{self.rule_type.value}|{self.version}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class ConstitutionalAmendment:
    """An amendment to the constitutional system."""
    amendment_id: str
    target_rule_id: Optional[str]  # None for new rules
    amendment_type: str  # "create", "modify", "repeal"
    proposed_changes: Dict[str, Any]
    justification: str
    proposed_by: str
    proposed_at: datetime
    voting_session_id: Optional[str] = None
    status: str = "proposed"
    community_feedback: List[Dict[str, Any]] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


class ConstitutionalStorage:
    """
    Constitutional rule storage and versioning system.
    Manages the storage, retrieval, and versioning of constitutional rules using LangGraph Store API.
    """

    def __init__(self, store_integration: MemoryStoreIntegration, postgres_persistence=None):
        """
        Initialize the Constitutional Storage system.
        
        Args:
            store_integration: Store API integration instance
            postgres_persistence: PostgreSQL persistence layer
        """
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.logger = logging.getLogger(f"{__name__}.ConstitutionalStorage")
        
        # In-memory cache for active rules
        self.active_rules = {}  # rule_id -> ConstitutionalRule
        self.rules_by_type = {}  # rule_type -> List[rule_id]
        
        # Version tracking
        self.rule_version_history = {}  # rule_id -> List[version_numbers]
        self.latest_versions = {}  # base_rule_id -> latest_version_number
        
        # Performance metrics
        self.metrics = {
            "rules_stored": 0,
            "rules_retrieved": 0,
            "versions_created": 0,
            "cache_hit_rate": 0.0
        }

    # =====================================================
    # Rule Storage and Retrieval
    # =====================================================

    async def store_rule(self, rule: ConstitutionalRule) -> bool:
        """
        Store a constitutional rule in the system.
        
        Args:
            rule: The constitutional rule to store
        
        Returns:
            bool: True if successfully stored
        """
        try:
            # Add rule hash for integrity
            rule.metadata["rule_hash"] = rule.get_rule_hash()
            rule.metadata["stored_at"] = datetime.now().isoformat()
            
            # Update in-memory cache
            self.active_rules[rule.rule_id] = rule
            
            # Update type index
            if rule.rule_type not in self.rules_by_type:
                self.rules_by_type[rule.rule_type] = []
            if rule.rule_id not in self.rules_by_type[rule.rule_type]:
                self.rules_by_type[rule.rule_type].append(rule.rule_id)
            
            # Update version tracking
            base_rule_id = rule.rule_id.split("_v")[0]  # Extract base ID
            if base_rule_id not in self.rule_version_history:
                self.rule_version_history[base_rule_id] = []
            if rule.version not in self.rule_version_history[base_rule_id]:
                self.rule_version_history[base_rule_id].append(rule.version)
            
            # Update latest version tracking
            if base_rule_id not in self.latest_versions or rule.version > self.latest_versions[base_rule_id]:
                self.latest_versions[base_rule_id] = rule.version
            
            # Store in Store API
            rule_data = asdict(rule)
            rule_data["created_at"] = rule.created_at.isoformat()
            rule_data["effective_date"] = rule.effective_date.isoformat()
            if rule.expiration_date:
                rule_data["expiration_date"] = rule.expiration_date.isoformat()
            rule_data["rule_type"] = rule.rule_type.value
            rule_data["status"] = rule.status.value
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"constitutional_rule_{rule.rule_id}",
                    rule_data
                )
            
            # Store in PostgreSQL for persistence
            if self.postgres_persistence:
                await self._store_rule_in_postgres(rule)
            
            # Update indices in Store API
            await self._update_rule_indices(rule)
            
            self.metrics["rules_stored"] += 1
            self.logger.info(f"Stored constitutional rule {rule.rule_id} (v{rule.version})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store constitutional rule {rule.rule_id}: {str(e)}")
            return False

    async def retrieve_rule(self, rule_id: str, version: Optional[int] = None) -> Optional[ConstitutionalRule]:
        """
        Retrieve a constitutional rule by ID and optional version.
        
        Args:
            rule_id: The rule ID to retrieve
            version: Optional specific version (defaults to latest)
        
        Returns:
            ConstitutionalRule if found, None otherwise
        """
        try:
            cache_hit = False
            
            # Try in-memory cache first
            if rule_id in self.active_rules and (version is None or self.active_rules[rule_id].version == version):
                cache_hit = True
                rule = self.active_rules[rule_id]
            else:
                # Try Store API
                rule = await self._retrieve_rule_from_store(rule_id, version)
                if rule:
                    self.active_rules[rule.rule_id] = rule  # Cache it
            
            # Update metrics
            self.metrics["rules_retrieved"] += 1
            if cache_hit:
                self.metrics["cache_hit_rate"] = (
                    self.metrics["cache_hit_rate"] * (self.metrics["rules_retrieved"] - 1) + 1.0
                ) / self.metrics["rules_retrieved"]
            else:
                self.metrics["cache_hit_rate"] = (
                    self.metrics["cache_hit_rate"] * (self.metrics["rules_retrieved"] - 1)
                ) / self.metrics["rules_retrieved"]
            
            if rule:
                self.logger.debug(f"Retrieved constitutional rule {rule_id} (v{rule.version})")
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve constitutional rule {rule_id}: {str(e)}")
            return None

    async def get_rules_by_type(self, rule_type: RuleType, active_only: bool = True) -> List[ConstitutionalRule]:
        """
        Get all rules of a specific type.
        
        Args:
            rule_type: The type of rules to retrieve
            active_only: Whether to return only active rules
        
        Returns:
            List of ConstitutionalRule objects
        """
        try:
            rules = []
            
            # Get rule IDs from type index
            rule_ids = self.rules_by_type.get(rule_type, [])
            
            # Retrieve each rule
            for rule_id in rule_ids:
                rule = await self.retrieve_rule(rule_id)
                if rule:
                    if not active_only or rule.status == RuleStatus.ACTIVE:
                        rules.append(rule)
            
            # Sort by precedence level (highest first)
            rules.sort(key=lambda x: x.precedence_level, reverse=True)
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Failed to get rules by type {rule_type}: {str(e)}")
            return []

    async def get_all_active_rules(self) -> Dict[str, ConstitutionalRule]:
        """
        Get all currently active constitutional rules.
        
        Returns:
            Dict mapping rule_id to ConstitutionalRule for all active rules
        """
        try:
            active_rules = {}
            
            # Check all cached rules
            for rule_id, rule in self.active_rules.items():
                if rule.status == RuleStatus.ACTIVE and datetime.now() >= rule.effective_date:
                    if rule.expiration_date is None or datetime.now() < rule.expiration_date:
                        active_rules[rule_id] = rule
            
            # Also check Store API for any rules not in cache
            if self.store_integration.store:
                all_rule_keys = await self.store_integration.store.asearch(
                    StoreNamespace.GOVERNANCE.value,
                    query="constitutional_rule_*"
                )
                
                for key in all_rule_keys:
                    rule_id = key.replace("constitutional_rule_", "")
                    if rule_id not in active_rules:
                        rule = await self.retrieve_rule(rule_id)
                        if rule and rule.status == RuleStatus.ACTIVE:
                            if datetime.now() >= rule.effective_date:
                                if rule.expiration_date is None or datetime.now() < rule.expiration_date:
                                    active_rules[rule_id] = rule
            
            return active_rules
            
        except Exception as e:
            self.logger.error(f"Failed to get all active rules: {str(e)}")
            return {}

    # =====================================================
    # Version Management
    # =====================================================

    async def create_rule_version(self, base_rule_id: str, updated_rule: ConstitutionalRule) -> str:
        """
        Create a new version of an existing rule.
        
        Args:
            base_rule_id: The base rule ID to version
            updated_rule: The updated rule content
        
        Returns:
            str: New versioned rule ID
        """
        try:
            # Get latest version number
            base_rule_id = base_rule_id.split("_v")[0]  # Remove existing version suffix
            latest_version = self.latest_versions.get(base_rule_id, 0)
            new_version = latest_version + 1
            
            # Create new versioned rule ID
            new_rule_id = f"{base_rule_id}_v{new_version}"
            
            # Update rule with new version info
            updated_rule.rule_id = new_rule_id
            updated_rule.version = new_version
            updated_rule.created_at = datetime.now()
            
            # Add to amendment history
            if base_rule_id in self.active_rules:
                old_rule = self.active_rules[base_rule_id]
                if old_rule.rule_id not in updated_rule.amendment_history:
                    updated_rule.amendment_history.append(old_rule.rule_id)
            
            # Store the new version
            success = await self.store_rule(updated_rule)
            if success:
                self.metrics["versions_created"] += 1
                self.logger.info(f"Created rule version {new_rule_id}")
                return new_rule_id
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to create rule version for {base_rule_id}: {str(e)}")
            return ""

    async def get_rule_history(self, base_rule_id: str) -> List[ConstitutionalRule]:
        """
        Get the version history of a rule.
        
        Args:
            base_rule_id: The base rule ID
        
        Returns:
            List of ConstitutionalRule objects in chronological order
        """
        try:
            base_rule_id = base_rule_id.split("_v")[0]  # Remove version suffix
            versions = self.rule_version_history.get(base_rule_id, [])
            
            rule_history = []
            for version in sorted(versions):
                versioned_id = f"{base_rule_id}_v{version}" if version > 1 else base_rule_id
                rule = await self.retrieve_rule(versioned_id, version)
                if rule:
                    rule_history.append(rule)
            
            return rule_history
            
        except Exception as e:
            self.logger.error(f"Failed to get rule history for {base_rule_id}: {str(e)}")
            return []

    async def deprecate_rule(self, rule_id: str, reason: str) -> bool:
        """
        Mark a rule as deprecated.
        
        Args:
            rule_id: The rule ID to deprecate
            reason: Reason for deprecation
        
        Returns:
            bool: True if successfully deprecated
        """
        try:
            rule = await self.retrieve_rule(rule_id)
            if not rule:
                return False
            
            rule.status = RuleStatus.DEPRECATED
            rule.metadata["deprecated_at"] = datetime.now().isoformat()
            rule.metadata["deprecation_reason"] = reason
            
            success = await self.store_rule(rule)
            if success:
                self.logger.info(f"Deprecated rule {rule_id}: {reason}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to deprecate rule {rule_id}: {str(e)}")
            return False

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _store_rule_in_postgres(self, rule: ConstitutionalRule) -> None:
        """Store rule in PostgreSQL for persistence."""
        if not self.postgres_persistence:
            return
        
        try:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO constitutional_rules (
                        rule_id, title, content, rule_type, status, version, created_at,
                        created_by, effective_date, expiration_date, precedence_level,
                        requires_supermajority, enforcement_mechanism, related_rules,
                        violation_penalties, amendment_history, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    ON CONFLICT (rule_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        status = EXCLUDED.status,
                        effective_date = EXCLUDED.effective_date,
                        expiration_date = EXCLUDED.expiration_date,
                        metadata = EXCLUDED.metadata
                """, 
                rule.rule_id, rule.title, rule.content, rule.rule_type.value, 
                rule.status.value, rule.version, rule.created_at, rule.created_by,
                rule.effective_date, rule.expiration_date, rule.precedence_level,
                rule.requires_supermajority, rule.enforcement_mechanism,
                json.dumps(rule.related_rules), json.dumps(rule.violation_penalties),
                json.dumps(rule.amendment_history), json.dumps(rule.metadata))
                
        except Exception as e:
            self.logger.error(f"Failed to store rule in PostgreSQL: {str(e)}")

    async def _retrieve_rule_from_store(self, rule_id: str, version: Optional[int] = None) -> Optional[ConstitutionalRule]:
        """Retrieve rule from Store API."""
        try:
            if not self.store_integration.store:
                return None
            
            rule_data = await self.store_integration.store.aget(
                StoreNamespace.GOVERNANCE.value,
                f"constitutional_rule_{rule_id}"
            )
            
            if not rule_data:
                return None
            
            # Check version match if specified
            if version is not None and rule_data.get("version", 1) != version:
                return None
            
            # Reconstruct ConstitutionalRule object
            rule = ConstitutionalRule(
                rule_id=rule_data["rule_id"],
                title=rule_data["title"],
                content=rule_data["content"],
                rule_type=RuleType(rule_data["rule_type"]),
                status=RuleStatus(rule_data["status"]),
                version=rule_data["version"],
                created_at=datetime.fromisoformat(rule_data["created_at"]),
                created_by=rule_data["created_by"],
                effective_date=datetime.fromisoformat(rule_data["effective_date"]),
                expiration_date=datetime.fromisoformat(rule_data["expiration_date"]) if rule_data.get("expiration_date") else None,
                precedence_level=rule_data.get("precedence_level", 100),
                requires_supermajority=rule_data.get("requires_supermajority", False),
                enforcement_mechanism=rule_data.get("enforcement_mechanism", "community_based"),
                related_rules=rule_data.get("related_rules", []),
                violation_penalties=rule_data.get("violation_penalties", {}),
                amendment_history=rule_data.get("amendment_history", []),
                metadata=rule_data.get("metadata", {})
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve rule {rule_id} from store: {str(e)}")
            return None

    async def _update_rule_indices(self, rule: ConstitutionalRule) -> None:
        """Update rule indices in Store API."""
        try:
            if not self.store_integration.store:
                return
            
            # Update type index
            type_index_key = f"rule_type_index_{rule.rule_type.value}"
            existing_index = await self.store_integration.store.aget(
                StoreNamespace.GOVERNANCE.value, type_index_key
            ) or []
            
            if rule.rule_id not in existing_index:
                existing_index.append(rule.rule_id)
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    type_index_key,
                    existing_index
                )
            
            # Update active rules index
            if rule.status == RuleStatus.ACTIVE:
                active_index_key = "active_rules_index"
                active_index = await self.store_integration.store.aget(
                    StoreNamespace.GOVERNANCE.value, active_index_key
                ) or []
                
                if rule.rule_id not in active_index:
                    active_index.append(rule.rule_id)
                    await self.store_integration.store.aput(
                        StoreNamespace.GOVERNANCE.value,
                        active_index_key,
                        active_index
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to update rule indices: {str(e)}")

    # =====================================================
    # System Health and Metrics
    # =====================================================

    async def get_storage_metrics(self) -> Dict[str, Any]:
        """Get constitutional storage system metrics."""
        try:
            total_rules = len(self.active_rules)
            active_rules = len([r for r in self.active_rules.values() if r.status == RuleStatus.ACTIVE])
            
            rule_type_distribution = {}
            for rule_type in RuleType:
                rule_type_distribution[rule_type.value] = len(self.rules_by_type.get(rule_type, []))
            
            return {
                "total_rules_stored": self.metrics["rules_stored"],
                "total_rules_retrieved": self.metrics["rules_retrieved"],
                "versions_created": self.metrics["versions_created"],
                "cache_hit_rate": self.metrics["cache_hit_rate"],
                "total_rules_in_memory": total_rules,
                "active_rules_count": active_rules,
                "rule_type_distribution": rule_type_distribution,
                "average_precedence_level": sum(r.precedence_level for r in self.active_rules.values()) / max(1, total_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage metrics: {str(e)}")
            return {}

    async def validate_rule_integrity(self, rule_id: str) -> bool:
        """Validate the integrity of a stored rule."""
        try:
            rule = await self.retrieve_rule(rule_id)
            if not rule:
                return False
            
            # Check rule hash
            expected_hash = rule.get_rule_hash()
            stored_hash = rule.metadata.get("rule_hash")
            
            if stored_hash and stored_hash != expected_hash:
                self.logger.warning(f"Rule integrity check failed for {rule_id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate rule integrity for {rule_id}: {str(e)}")
            return False


# Helper functions
def create_constitutional_storage(store_integration: MemoryStoreIntegration, 
                                postgres_persistence=None) -> ConstitutionalStorage:
    """Create a ConstitutionalStorage instance."""
    return ConstitutionalStorage(store_integration, postgres_persistence)


def create_constitutional_rule(title: str, content: str, rule_type: RuleType,
                             created_by: str, **kwargs) -> ConstitutionalRule:
    """Helper function to create a constitutional rule."""
    rule_id = kwargs.get("rule_id", str(uuid.uuid4())[:8])
    
    return ConstitutionalRule(
        rule_id=rule_id,
        title=title,
        content=content,
        rule_type=rule_type,
        status=RuleStatus.PROPOSED,
        version=1,
        created_at=datetime.now(),
        created_by=created_by,
        effective_date=kwargs.get("effective_date", datetime.now()),
        **{k: v for k, v in kwargs.items() if k not in ["rule_id", "effective_date"]}
    )