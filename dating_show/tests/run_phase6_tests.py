#!/usr/bin/env python3
"""
Phase 6 Integration Test Runner
Comprehensive test runner for all Phase 6 integration components
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dating_show.tests.test_integration_phase6 import run_integration_tests


def main():
    """Main test runner function"""
    print("=" * 70)
    print("DATING SHOW PHASE 6 INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test environment info
    print("Test Environment:")
    print(f"  Python version: {sys.version}")
    print(f"  Project root: {project_root}")
    print(f"  Working directory: {os.getcwd()}")
    print()
    
    # Run tests
    print("Running Phase 6 Integration Tests...")
    print("-" * 40)
    
    start_time = time.time()
    success = run_integration_tests()
    end_time = time.time()
    
    print()
    print("-" * 40)
    print(f"Test execution completed in {end_time - start_time:.2f} seconds")
    
    # Summary
    if success:
        print("✅ ALL TESTS PASSED")
        print()
        print("Phase 6 Integration Status:")
        print("  ✅ Database Service - READY")
        print("  ✅ Enhanced Frontend Bridge - READY") 
        print("  ✅ Orchestration Service - READY")
        print("  ✅ PIANO Integration - READY")
        print("  ✅ Full Integration - READY")
        print()
        print("🎉 PHASE 6 IMPLEMENTATION COMPLETE!")
        print("   The dating show frontend integration is ready for deployment.")
        
        # Create success report
        report = {
            "test_status": "PASSED",
            "execution_time": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database_service": "READY",
                "enhanced_bridge": "READY", 
                "orchestration": "READY",
                "piano_integration": "READY",
                "full_integration": "READY"
            },
            "next_steps": [
                "Deploy to production environment",
                "Configure database connections", 
                "Start Django frontend server",
                "Launch dating show simulation",
                "Monitor system health and performance"
            ]
        }
        
        report_path = project_root / "PHASE6_INTEGRATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Integration report saved to: {report_path}")
        
        return 0
        
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please review test output above and fix any issues before proceeding.")
        print("Integration components may not be fully ready for production.")
        
        return 1


if __name__ == "__main__":
    exit(main())