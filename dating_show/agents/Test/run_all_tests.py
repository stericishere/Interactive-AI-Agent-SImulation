#!/usr/bin/env python3
"""
Comprehensive Test Runner for Enhanced PIANO Memory Architecture
Executes all test suites and generates unified coverage and quality reports.
"""

import sys
import os
import time
import subprocess
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')


class UnifiedTestRunner:
    """Unified test runner for all PIANO memory architecture test suites"""
    
    def __init__(self):
        self.test_suites = [
            {
                'name': 'Memory Systems Tests',
                'file': 'test_memory_systems.py',
                'description': 'Core functionality tests for all memory components',
                'category': 'functional'
            },
            {
                'name': 'Integration Tests',
                'file': 'test_integration.py',
                'description': 'System integration and coordination tests',
                'category': 'integration'
            },
            {
                'name': 'Simple Integration Tests',
                'file': 'test_integration_simple.py',
                'description': 'Simplified integration test suite',
                'category': 'integration'
            },
            {
                'name': 'Comprehensive Coverage Tests',
                'file': 'test_comprehensive_coverage.py',
                'description': 'Tests for previously untested methods and edge cases',
                'category': 'coverage'
            },
            {
                'name': 'Stress and Load Tests',
                'file': 'test_stress_and_load.py',
                'description': 'High-load, stress testing, and resource constraints',
                'category': 'performance'
            },
            {
                'name': 'Security and Validation Tests',
                'file': 'test_security_and_validation.py',
                'description': 'Security, input validation, and vulnerability testing',
                'category': 'security'
            },
            {
                'name': 'Compatibility and Regression Tests',
                'file': 'test_compatibility_and_regression.py',
                'description': 'Backward compatibility and performance regression testing',
                'category': 'compatibility'
            }
        ]
        
        self.results = {}
        self.overall_stats = {
            'total_suites': 0,
            'passed_suites': 0,
            'failed_suites': 0,
            'total_execution_time': 0,
            'coverage_improvement': 0,
            'vulnerabilities_found': 0,
            'performance_regressions': 0
        }
    
    def run_test_suite(self, suite: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
        """Run a single test suite and capture results"""
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING: {suite['name']}")
        print(f"📁 File: {suite['file']}")
        print(f"📝 Description: {suite['description']}")
        print(f"{'='*70}")
        
        suite_result = {
            'name': suite['name'],
            'file': suite['file'],
            'category': suite['category'],
            'start_time': datetime.now().isoformat(),
            'success': False,
            'execution_time': 0,
            'output': '',
            'error_output': '',
            'exit_code': 1,
            'stats': {}
        }
        
        try:
            start_time = time.perf_counter()
            
            # Run the test suite
            result = subprocess.run(
                [sys.executable, suite['file']],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per suite
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Capture results
            suite_result.update({
                'success': result.returncode == 0,
                'execution_time': execution_time,
                'output': result.stdout,
                'error_output': result.stderr,
                'exit_code': result.returncode,
                'end_time': datetime.now().isoformat()
            })
            
            # Parse output for statistics
            suite_result['stats'] = self._parse_test_output(result.stdout, suite['category'])
            
            # Print summary
            status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
            print(f"\n{status} - {suite['name']} ({execution_time:.2f}s)")
            
            if result.returncode != 0:
                print(f"Error output: {result.stderr}")
            
            return result.returncode == 0, suite_result
        
        except subprocess.TimeoutExpired:
            suite_result.update({
                'success': False,
                'error_output': 'Test suite timed out after 5 minutes',
                'execution_time': 300,
                'end_time': datetime.now().isoformat()
            })
            print(f"⏰ TIMEOUT - {suite['name']} (exceeded 5 minutes)")
            return False, suite_result
        
        except Exception as e:
            suite_result.update({
                'success': False,
                'error_output': str(e),
                'execution_time': 0,
                'end_time': datetime.now().isoformat()
            })
            print(f"💥 ERROR - {suite['name']}: {str(e)}")
            return False, suite_result
    
    def _parse_test_output(self, output: str, category: str) -> Dict[str, Any]:
        """Parse test output to extract statistics"""
        stats = {}
        
        try:
            lines = output.split('\n')
            
            # Look for common patterns
            for line in lines:
                line = line.strip()
                
                # Test summary patterns
                if 'TEST SUMMARY:' in line or 'SUMMARY:' in line:
                    # Extract pass/fail counts
                    if '/' in line and 'passed' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '/' in part and part.replace('/', '').replace('.', '').isdigit():
                                passed, total = part.split('/')
                                stats['tests_passed'] = int(passed)
                                stats['tests_total'] = int(total)
                                break
                
                # Coverage patterns
                if 'COVERAGE:' in line and '%' in line:
                    coverage_match = line.split('%')[0].split()[-1]
                    try:
                        stats['coverage_percentage'] = float(coverage_match)
                    except:
                        pass
                
                # Performance patterns
                if 'ops/sec' in line:
                    try:
                        ops_per_sec = line.split('ops/sec')[0].split('(')[-1].strip()
                        stats['operations_per_second'] = float(ops_per_sec)
                    except:
                        pass
                
                # Security/vulnerability patterns
                if 'VULNERABILITIES:' in line:
                    try:
                        vuln_count = line.split('VULNERABILITIES:')[1].strip().split()[0]
                        stats['vulnerabilities_found'] = int(vuln_count)
                    except:
                        pass
                
                # Performance regression patterns
                if 'REGRESSIONS:' in line:
                    try:
                        reg_count = line.split('REGRESSIONS:')[1].strip().split()[0]
                        stats['performance_regressions'] = int(reg_count)
                    except:
                        pass
        
        except Exception as e:
            stats['parse_error'] = str(e)
        
        return stats
    
    def generate_unified_report(self) -> str:
        """Generate unified test report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            "🧪 Enhanced PIANO Memory Architecture - Unified Test Report",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Test Suites: {self.overall_stats['total_suites']}",
            f"Execution Time: {self.overall_stats['total_execution_time']:.2f} seconds",
            ""
        ])
        
        # Executive Summary
        success_rate = (self.overall_stats['passed_suites'] / 
                       self.overall_stats['total_suites'] * 100) if self.overall_stats['total_suites'] > 0 else 0
        
        report_lines.extend([
            "📊 EXECUTIVE SUMMARY",
            "-" * 50,
            f"✅ Passed Suites: {self.overall_stats['passed_suites']}/{self.overall_stats['total_suites']} ({success_rate:.1f}%)",
            f"❌ Failed Suites: {self.overall_stats['failed_suites']}",
            f"📈 Coverage Improvement: {self.overall_stats['coverage_improvement']:.1f}%",
            f"🚨 Vulnerabilities Found: {self.overall_stats['vulnerabilities_found']}",
            f"📉 Performance Regressions: {self.overall_stats['performance_regressions']}",
            ""
        ])
        
        # Quality Assessment
        if success_rate >= 95:
            quality_status = "🎉 EXCELLENT - Production Ready"
        elif success_rate >= 85:
            quality_status = "✅ GOOD - Minor Issues"
        elif success_rate >= 70:
            quality_status = "⚠️  FAIR - Needs Attention"
        else:
            quality_status = "❌ POOR - Major Issues"
        
        report_lines.extend([
            f"🏆 OVERALL QUALITY: {quality_status}",
            ""
        ])
        
        # Suite-by-Suite Results
        report_lines.extend([
            "📋 DETAILED RESULTS BY SUITE",
            "-" * 50
        ])
        
        categories = {}
        for suite_name, result in self.results.items():
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, suites in categories.items():
            report_lines.append(f"\n🔧 {category.upper()} TESTS:")
            
            for suite in suites:
                status = "✅" if suite['success'] else "❌"
                execution_time = suite.get('execution_time', 0)
                
                report_lines.append(f"  {status} {suite['name']} ({execution_time:.2f}s)")
                
                # Add suite statistics
                stats = suite.get('stats', {})
                if stats:
                    if 'tests_passed' in stats and 'tests_total' in stats:
                        test_rate = (stats['tests_passed'] / stats['tests_total'] * 100) if stats['tests_total'] > 0 else 0
                        report_lines.append(f"      Tests: {stats['tests_passed']}/{stats['tests_total']} ({test_rate:.1f}%)")
                    
                    if 'coverage_percentage' in stats:
                        report_lines.append(f"      Coverage: {stats['coverage_percentage']:.1f}%")
                    
                    if 'vulnerabilities_found' in stats:
                        report_lines.append(f"      Vulnerabilities: {stats['vulnerabilities_found']}")
                    
                    if 'performance_regressions' in stats:
                        report_lines.append(f"      Regressions: {stats['performance_regressions']}")
                
                # Add error details for failed suites
                if not suite['success'] and suite.get('error_output'):
                    error_preview = suite['error_output'][:200]
                    if len(suite['error_output']) > 200:
                        error_preview += "..."
                    report_lines.append(f"      Error: {error_preview}")
        
        # Recommendations
        report_lines.extend([
            "",
            "💡 RECOMMENDATIONS",
            "-" * 50
        ])
        
        recommendations = []
        
        if self.overall_stats['failed_suites'] > 0:
            recommendations.append("• Fix failing test suites before production deployment")
        
        if self.overall_stats['vulnerabilities_found'] > 0:
            recommendations.append("• Address security vulnerabilities immediately")
        
        if self.overall_stats['performance_regressions'] > 0:
            recommendations.append("• Investigate and fix performance regressions")
        
        if self.overall_stats['coverage_improvement'] < 5:
            recommendations.append("• Consider adding more comprehensive test coverage")
        
        if success_rate < 85:
            recommendations.append("• Implement continuous integration to catch issues early")
        
        if not recommendations:
            recommendations.append("✅ Excellent test results! System is production-ready.")
        
        report_lines.extend(recommendations)
        
        # Next Steps
        report_lines.extend([
            "",
            "🚀 NEXT STEPS",
            "-" * 50,
            "1. Review and address any failing tests",
            "2. Fix security vulnerabilities if found",
            "3. Optimize performance regressions",
            "4. Update documentation based on test results",
            "5. Consider implementing automated CI/CD pipeline",
            "6. Schedule regular regression testing",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def run_all_tests(self) -> bool:
        """Run all test suites and generate unified report"""
        print("🧪 Enhanced PIANO Memory Architecture - Comprehensive Test Execution")
        print("=" * 70)
        print(f"Starting execution of {len(self.test_suites)} test suites...")
        
        start_time = time.perf_counter()
        
        self.overall_stats['total_suites'] = len(self.test_suites)
        
        # Run each test suite
        for suite in self.test_suites:
            success, result = self.run_test_suite(suite)
            self.results[suite['name']] = result
            
            if success:
                self.overall_stats['passed_suites'] += 1
            else:
                self.overall_stats['failed_suites'] += 1
            
            # Aggregate statistics
            stats = result.get('stats', {})
            if 'vulnerabilities_found' in stats:
                self.overall_stats['vulnerabilities_found'] += stats['vulnerabilities_found']
            
            if 'performance_regressions' in stats:
                self.overall_stats['performance_regressions'] += stats['performance_regressions']
        
        end_time = time.perf_counter()
        self.overall_stats['total_execution_time'] = end_time - start_time
        
        # Generate and display report
        report = self.generate_unified_report()
        print("\n" + report)
        
        # Save report to file
        report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Also save JSON results for programmatic access
        json_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w') as f:
            json.dump({
                'overall_stats': self.overall_stats,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"📊 Machine-readable results saved to: {json_filename}")
        
        # Return overall success
        return self.overall_stats['failed_suites'] == 0


def main():
    """Main execution function"""
    runner = UnifiedTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 ALL TEST SUITES PASSED! Memory architecture is production-ready.")
        return 0
    else:
        print(f"\n💥 {runner.overall_stats['failed_suites']} test suites failed. Review results above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)