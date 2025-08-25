#!/usr/bin/env python3
"""
End-to-End System Validation Script
Epic 4: Production Deployment & Validation

Comprehensive validation of the Dating Show Frontend Service
with unified architecture integration.
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any
import websockets
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class ValidationResult:
    def __init__(self, test_name: str, success: bool, message: str, details: Dict = None):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

class SystemValidator:
    """Comprehensive system validation for production deployment"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[ValidationResult] = []
        self.session = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_result(self, result: ValidationResult):
        """Log validation result with colored output"""
        self.results.append(result)
        
        icon = f"{Colors.GREEN}✅{Colors.END}" if result.success else f"{Colors.RED}❌{Colors.END}"
        color = Colors.GREEN if result.success else Colors.RED
        
        print(f"{icon} {color}{result.test_name}{Colors.END}: {result.message}")
        
        if result.details:
            for key, value in result.details.items():
                print(f"   {Colors.CYAN}{key}{Colors.END}: {value}")
    
    async def validate_service_startup(self) -> ValidationResult:
        """Test 1: Service startup and basic connectivity"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return ValidationResult(
                        "Service Startup",
                        True,
                        f"Service running on {self.base_url}",
                        {"status": data.get("overall_status"), "version": data.get("version")}
                    )
                else:
                    return ValidationResult(
                        "Service Startup",
                        False,
                        f"Service responded with status {response.status}",
                        {"status_code": response.status}
                    )
        except Exception as e:
            return ValidationResult(
                "Service Startup",
                False,
                f"Failed to connect to service: {str(e)}",
                {"error": str(e)}
            )
    
    async def validate_health_checks(self) -> ValidationResult:
        """Test 2: Comprehensive health check validation"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    total_checks = data.get("summary", {}).get("total_checks", 0)
                    healthy_checks = data.get("summary", {}).get("healthy", 0)
                    success_rate = data.get("summary", {}).get("success_rate", 0)
                    
                    if success_rate >= 80:  # Allow some degraded checks
                        return ValidationResult(
                            "Health Checks",
                            True,
                            f"Health checks passing: {healthy_checks}/{total_checks} ({success_rate}%)",
                            {
                                "total_checks": total_checks,
                                "healthy": healthy_checks,
                                "success_rate": f"{success_rate}%",
                                "overall_status": data.get("overall_status")
                            }
                        )
                    else:
                        failed_checks = [
                            name for name, check in data.get("checks", {}).items()
                            if check.get("status") == "unhealthy"
                        ]
                        return ValidationResult(
                            "Health Checks",
                            False,
                            f"Too many health check failures: {success_rate}%",
                            {"failed_checks": failed_checks}
                        )
                else:
                    return ValidationResult(
                        "Health Checks",
                        False,
                        f"Health endpoint returned {response.status}",
                        {"status_code": response.status}
                    )
        except Exception as e:
            return ValidationResult(
                "Health Checks",
                False,
                f"Health check validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def validate_static_files(self) -> ValidationResult:
        """Test 3: Static file serving"""
        static_files = [
            "/static/js/websocket-client.js",
            "/static/js/agent-visualizer.js",
            "/static/js/social-network-viz.js",
            "/static/js/performance-dashboard.js",
            "/static/js/dating-show-app.js"
        ]
        
        successful_loads = 0
        failed_files = []
        
        for file_path in static_files:
            try:
                async with self.session.get(f"{self.base_url}{file_path}") as response:
                    if response.status == 200:
                        content = await response.text()
                        if len(content) > 100:  # Basic content validation
                            successful_loads += 1
                        else:
                            failed_files.append(f"{file_path} (empty/too small)")
                    else:
                        failed_files.append(f"{file_path} ({response.status})")
            except Exception as e:
                failed_files.append(f"{file_path} (error: {str(e)})")
        
        if successful_loads == len(static_files):
            return ValidationResult(
                "Static Files",
                True,
                f"All {len(static_files)} JavaScript files loaded successfully",
                {"loaded_files": successful_loads}
            )
        else:
            return ValidationResult(
                "Static Files",
                False,
                f"Only {successful_loads}/{len(static_files)} files loaded",
                {"failed_files": failed_files}
            )
    
    async def validate_dashboard_rendering(self) -> ValidationResult:
        """Test 4: Dashboard template rendering"""
        try:
            async with self.session.get(f"{self.base_url}/dashboard") as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Check for key components in the rendered HTML
                    required_elements = [
                        "dating-show-app",
                        "websocket-client.js",
                        "agent-visualizer.js",
                        "social-network-viz.js",
                        "performance-dashboard.js",
                        "dating-show-app.js",
                        "d3js.org/d3.v7.min.js"
                    ]
                    
                    missing_elements = []
                    for element in required_elements:
                        if element not in content:
                            missing_elements.append(element)
                    
                    if not missing_elements:
                        return ValidationResult(
                            "Dashboard Rendering",
                            True,
                            "Dashboard template rendered with all required components",
                            {"content_length": len(content)}
                        )
                    else:
                        return ValidationResult(
                            "Dashboard Rendering",
                            False,
                            f"Missing required elements in dashboard",
                            {"missing_elements": missing_elements}
                        )
                else:
                    return ValidationResult(
                        "Dashboard Rendering",
                        False,
                        f"Dashboard returned status {response.status}",
                        {"status_code": response.status}
                    )
        except Exception as e:
            return ValidationResult(
                "Dashboard Rendering",
                False,
                f"Dashboard rendering test failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def validate_websocket_endpoints(self) -> ValidationResult:
        """Test 5: WebSocket endpoints functionality"""
        websocket_endpoints = [
            "/api/ws/agents/general",
            "/api/ws/agents/test_room",
            "/api/ws/system"
        ]
        
        successful_connections = 0
        failed_connections = []
        
        for endpoint in websocket_endpoints:
            try:
                ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
                async with websockets.connect(f"{ws_url}{endpoint}") as websocket:
                    # Send a ping message
                    await websocket.send(json.dumps({"type": "ping"}))
                    
                    # Wait for response with timeout
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "connection_established" or data.get("type") == "pong":
                        successful_connections += 1
                    else:
                        failed_connections.append(f"{endpoint} (unexpected response: {data.get('type')})")
                        
            except asyncio.TimeoutError:
                failed_connections.append(f"{endpoint} (timeout)")
            except Exception as e:
                failed_connections.append(f"{endpoint} (error: {str(e)})")
        
        if successful_connections == len(websocket_endpoints):
            return ValidationResult(
                "WebSocket Endpoints",
                True,
                f"All {len(websocket_endpoints)} WebSocket endpoints operational",
                {"successful_connections": successful_connections}
            )
        else:
            return ValidationResult(
                "WebSocket Endpoints",
                False,
                f"Only {successful_connections}/{len(websocket_endpoints)} WebSocket endpoints working",
                {"failed_connections": failed_connections}
            )
    
    async def validate_api_endpoints(self) -> ValidationResult:
        """Test 6: REST API endpoints"""
        api_endpoints = [
            ("/api/simulation/state", "GET"),
            ("/api/agents", "GET"),
            ("/health", "GET"),
            ("/health/websocket_support", "GET")
        ]
        
        successful_calls = 0
        failed_calls = []
        
        for endpoint, method in api_endpoints:
            try:
                if method == "GET":
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        if 200 <= response.status < 300:
                            successful_calls += 1
                        else:
                            failed_calls.append(f"{method} {endpoint} ({response.status})")
            except Exception as e:
                failed_calls.append(f"{method} {endpoint} (error: {str(e)})")
        
        if successful_calls == len(api_endpoints):
            return ValidationResult(
                "API Endpoints",
                True,
                f"All {len(api_endpoints)} API endpoints responding",
                {"successful_calls": successful_calls}
            )
        else:
            return ValidationResult(
                "API Endpoints",
                False,
                f"Only {successful_calls}/{len(api_endpoints)} API endpoints working",
                {"failed_calls": failed_calls}
            )
    
    async def validate_unified_architecture(self) -> ValidationResult:
        """Test 7: Unified architecture integration"""
        try:
            async with self.session.get(f"{self.base_url}/health/unified_architecture") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "healthy":
                        return ValidationResult(
                            "Unified Architecture",
                            True,
                            "Unified architecture components operational",
                            {
                                "status": data.get("status"),
                                "message": data.get("message"),
                                "details": data.get("details", {})
                            }
                        )
                    elif data.get("status") == "degraded":
                        return ValidationResult(
                            "Unified Architecture",
                            True,  # Still considered passing in fallback mode
                            "Unified architecture in fallback mode",
                            {"status": data.get("status"), "message": data.get("message")}
                        )
                    else:
                        return ValidationResult(
                            "Unified Architecture",
                            False,
                            f"Unified architecture unhealthy: {data.get('message')}",
                            {"status": data.get("status")}
                        )
                else:
                    return ValidationResult(
                        "Unified Architecture",
                        False,
                        f"Health check returned {response.status}",
                        {"status_code": response.status}
                    )
        except Exception as e:
            return ValidationResult(
                "Unified Architecture",
                False,
                f"Unified architecture validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def validate_performance_metrics(self) -> ValidationResult:
        """Test 8: Performance metrics and monitoring"""
        try:
            start_time = time.time()
            
            # Test response time for key endpoints
            async with self.session.get(f"{self.base_url}/health") as response:
                health_response_time = time.time() - start_time
                
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/dashboard") as response:
                dashboard_response_time = time.time() - start_time
            
            # Check if response times are acceptable
            health_acceptable = health_response_time < 1.0  # 1 second
            dashboard_acceptable = dashboard_response_time < 3.0  # 3 seconds
            
            if health_acceptable and dashboard_acceptable:
                return ValidationResult(
                    "Performance Metrics",
                    True,
                    "Response times within acceptable limits",
                    {
                        "health_response_time": f"{health_response_time:.2f}s",
                        "dashboard_response_time": f"{dashboard_response_time:.2f}s"
                    }
                )
            else:
                return ValidationResult(
                    "Performance Metrics",
                    False,
                    "Response times exceed acceptable limits",
                    {
                        "health_response_time": f"{health_response_time:.2f}s",
                        "dashboard_response_time": f"{dashboard_response_time:.2f}s",
                        "health_acceptable": health_acceptable,
                        "dashboard_acceptable": dashboard_acceptable
                    }
                )
        except Exception as e:
            return ValidationResult(
                "Performance Metrics",
                False,
                f"Performance validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def run_all_validations(self):
        """Run all validation tests"""
        print(f"{Colors.BOLD}{Colors.BLUE}🚀 Dating Show Frontend Service - End-to-End Validation{Colors.END}")
        print(f"{Colors.CYAN}Epic 4: Production Deployment & Validation{Colors.END}")
        print(f"Testing service at: {Colors.YELLOW}{self.base_url}{Colors.END}")
        print("-" * 70)
        
        validations = [
            self.validate_service_startup(),
            self.validate_health_checks(),
            self.validate_static_files(),
            self.validate_dashboard_rendering(),
            self.validate_websocket_endpoints(),
            self.validate_api_endpoints(),
            self.validate_unified_architecture(),
            self.validate_performance_metrics()
        ]
        
        for validation in validations:
            result = await validation
            self.log_result(result)
        
        # Summary
        print("-" * 70)
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"{Colors.BOLD}📊 Validation Summary{Colors.END}")
        print(f"Total Tests: {total_tests}")
        print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}Failed: {failed_tests}{Colors.END}")
        print(f"Success Rate: {Colors.GREEN if success_rate >= 90 else Colors.YELLOW if success_rate >= 75 else Colors.RED}{success_rate:.1f}%{Colors.END}")
        
        if success_rate >= 90:
            print(f"\n{Colors.GREEN}🎉 System validation PASSED! Ready for production deployment.{Colors.END}")
        elif success_rate >= 75:
            print(f"\n{Colors.YELLOW}⚠️  System validation PARTIAL. Review failed tests before production.{Colors.END}")
        else:
            print(f"\n{Colors.RED}❌ System validation FAILED. Address critical issues before deployment.{Colors.END}")
        
        # List failed tests
        if failed_tests > 0:
            print(f"\n{Colors.RED}Failed Tests:{Colors.END}")
            for result in self.results:
                if not result.success:
                    print(f"  • {result.test_name}: {result.message}")

async def main():
    """Main validation execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dating Show Frontend Service Validation")
    parser.add_argument("--url", default="http://localhost:8001", 
                       help="Base URL of the service (default: http://localhost:8001)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    async with SystemValidator(args.url) as validator:
        await validator.run_all_validations()
        
        # Exit with appropriate code
        total_tests = len(validator.results)
        passed_tests = sum(1 for r in validator.results if r.success)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate >= 90:
            sys.exit(0)  # Success
        elif success_rate >= 75:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Failure

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(3)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {str(e)}{Colors.END}")
        sys.exit(4)