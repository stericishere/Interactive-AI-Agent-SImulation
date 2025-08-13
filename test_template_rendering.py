#!/usr/bin/env python3
"""
Template Rendering Test Suite
Tests for Phase 5: Enhanced UI Templates and Components
"""

import os
import sys
import re
from pathlib import Path


def test_template_files_exist():
    """Test that all required template files exist"""
    template_dir = Path("environment/frontend_server/templates")
    
    required_templates = [
        "base_enhanced.html",
        "dating_show/main_dashboard.html"
    ]
    
    missing_templates = []
    existing_templates = []
    
    for template in required_templates:
        template_path = template_dir / template
        if template_path.exists():
            existing_templates.append(template)
        else:
            missing_templates.append(template)
    
    if missing_templates:
        print(f"❌ Missing templates: {missing_templates}")
        return False
    else:
        print(f"✅ All required templates exist ({len(existing_templates)} files)")
        return True


def test_base_template_structure():
    """Test base enhanced template structure"""
    try:
        template_path = Path("environment/frontend_server/templates/base_enhanced.html")
        
        if not template_path.exists():
            print("❌ Base enhanced template not found")
            return False
            
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for essential HTML structure
        required_elements = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<body>',
            '<nav class="navbar',
            'Bootstrap',
            'dating_show_theme.css',
            'WebSocket',
            '{% block content %}',
            '{% block extra_js %}'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Base template missing elements: {missing_elements}")
            return False
        
        # Test for modern features
        modern_features = [
            'Bootstrap 5',
            'WebSocket',
            'Font Awesome',
            'Chart.js',
            'D3.js',
            'vis-network'
        ]
        
        found_features = []
        for feature in modern_features:
            if feature.lower() in content.lower():
                found_features.append(feature)
        
        print(f"✅ Base template structure valid (found {len(found_features)}/{len(modern_features)} modern features)")
        return True
        
    except Exception as e:
        print(f"❌ Base template test failed: {e}")
        return False


def test_dashboard_template_structure():
    """Test main dashboard template structure"""
    try:
        template_path = Path("environment/frontend_server/templates/dating_show/main_dashboard.html")
        
        if not template_path.exists():
            print("❌ Dashboard template not found")
            return False
            
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for dashboard-specific elements
        required_elements = [
            '{% extends "base_enhanced.html" %}',
            'simulation-controls',
            'agents-container',
            'agent-card-template',
            'agent-pagination',
            'filter-container',
            'WebSocket',
            'loadAgents',
            'renderAgents'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Dashboard template missing elements: {missing_elements}")
            return False
        
        # Test for interactive features
        interactive_features = [
            'onclick=',
            'addEventListener',
            'getElementById',
            'querySelector',
            'WebSocket',
            'fetch(',
            'pagination'
        ]
        
        found_interactive = sum(1 for feature in interactive_features if feature in content)
        
        print(f"✅ Dashboard template structure valid (found {found_interactive}/{len(interactive_features)} interactive features)")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard template test failed: {e}")
        return False


def test_css_files_exist():
    """Test that required CSS files exist"""
    css_dir = Path("environment/frontend_server/static_dirs/dating_show/css")
    
    required_css_files = [
        "dating_show_theme.css",
        "agent_cards.css"
    ]
    
    missing_css = []
    existing_css = []
    
    for css_file in required_css_files:
        css_path = css_dir / css_file
        if css_path.exists():
            existing_css.append(css_file)
        else:
            missing_css.append(css_file)
    
    if missing_css:
        print(f"❌ Missing CSS files: {missing_css}")
        return False
    else:
        print(f"✅ All required CSS files exist ({len(existing_css)} files)")
        return True


def test_css_theme_structure():
    """Test CSS theme structure"""
    try:
        css_path = Path("environment/frontend_server/static_dirs/dating_show/css/dating_show_theme.css")
        
        if not css_path.exists():
            print("❌ Theme CSS file not found")
            return False
            
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for CSS structure
        required_css_elements = [
            ':root {',
            '--primary-color:',
            '--secondary-color:',
            '.agent-card',
            '.btn',
            '.navbar',
            '@media',
            'animation',
            'gradient'
        ]
        
        missing_css = []
        for element in required_css_elements:
            if element not in content:
                missing_css.append(element)
        
        if missing_css:
            print(f"❌ Theme CSS missing elements: {missing_css}")
            return False
        
        # Count CSS rules and animations
        css_rules = len(re.findall(r'[^{}]*{[^{}]*}', content))
        animations = len(re.findall(r'@keyframes', content))
        media_queries = len(re.findall(r'@media', content))
        
        print(f"✅ Theme CSS structure valid ({css_rules} rules, {animations} animations, {media_queries} media queries)")
        return True
        
    except Exception as e:
        print(f"❌ Theme CSS test failed: {e}")
        return False


def test_agent_cards_css():
    """Test agent cards CSS structure"""
    try:
        css_path = Path("environment/frontend_server/static_dirs/dating_show/css/agent_cards.css")
        
        if not css_path.exists():
            print("❌ Agent cards CSS file not found")
            return False
            
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for agent card specific styles
        required_elements = [
            '.agent-grid',
            '.agent-card',
            '.agent-avatar',
            '.agent-name',
            '.agent-status',
            '.skills-section',
            '@media',
            'transition:',
            'hover'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Agent cards CSS missing elements: {missing_elements}")
            return False
        
        # Test for responsive design
        responsive_features = [
            '@media (max-width: 1200px)',
            '@media (max-width: 768px)',
            '@media (max-width: 576px)',
            'grid-template-columns',
            'flex-direction'
        ]
        
        responsive_count = sum(1 for feature in responsive_features if feature in content)
        
        print(f"✅ Agent cards CSS structure valid (found {responsive_count}/{len(responsive_features)} responsive features)")
        return True
        
    except Exception as e:
        print(f"❌ Agent cards CSS test failed: {e}")
        return False


def test_javascript_functionality():
    """Test JavaScript functionality in templates"""
    try:
        template_path = Path("environment/frontend_server/templates/dating_show/main_dashboard.html")
        
        if not template_path.exists():
            print("❌ Dashboard template not found for JS testing")
            return False
            
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for essential JavaScript functions
        required_js_functions = [
            'loadAgents',
            'renderAgents',
            'applyFilters',
            'controlSimulation',
            'handleAgentUpdate',
            'WebSocket',
            'addEventListener',
            'fetch('
        ]
        
        missing_js = []
        for func in required_js_functions:
            if func not in content:
                missing_js.append(func)
        
        if missing_js:
            print(f"❌ Dashboard template missing JS functions: {missing_js}")
            return False
        
        # Test for modern JavaScript features
        modern_js_features = [
            'async function',
            'await ',
            'fetch(',
            'JSON.parse',
            'querySelector',
            'addEventListener',
            'template.content.cloneNode'
        ]
        
        modern_count = sum(1 for feature in modern_js_features if feature in content)
        
        print(f"✅ JavaScript functionality valid (found {modern_count}/{len(modern_js_features)} modern JS features)")
        return True
        
    except Exception as e:
        print(f"❌ JavaScript functionality test failed: {e}")
        return False


def test_accessibility_features():
    """Test accessibility features in templates"""
    try:
        template_path = Path("environment/frontend_server/templates/base_enhanced.html")
        
        if not template_path.exists():
            print("❌ Base template not found for accessibility testing")
            return False
            
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test for accessibility features
        accessibility_features = [
            'lang="en"',
            'alt=',
            'aria-',
            'role=',
            'tabindex',
            'for=',
            'id=',
            '<label'
        ]
        
        found_features = sum(1 for feature in accessibility_features if feature in content)
        
        if found_features < 3:  # Minimum accessibility features
            print(f"❌ Insufficient accessibility features found ({found_features})")
            return False
        
        print(f"✅ Accessibility features present (found {found_features}/{len(accessibility_features)} features)")
        return True
        
    except Exception as e:
        print(f"❌ Accessibility test failed: {e}")
        return False


def test_template_performance():
    """Test template file sizes for performance"""
    try:
        template_files = [
            "environment/frontend_server/templates/base_enhanced.html",
            "environment/frontend_server/templates/dating_show/main_dashboard.html"
        ]
        
        css_files = [
            "environment/frontend_server/static_dirs/dating_show/css/dating_show_theme.css",
            "environment/frontend_server/static_dirs/dating_show/css/agent_cards.css"
        ]
        
        total_template_size = 0
        total_css_size = 0
        
        for template_file in template_files:
            path = Path(template_file)
            if path.exists():
                total_template_size += path.stat().st_size
        
        for css_file in css_files:
            path = Path(css_file)
            if path.exists():
                total_css_size += path.stat().st_size
        
        # Templates should be reasonable size (not too large)
        max_template_size = 200 * 1024  # 200KB total for templates
        max_css_size = 100 * 1024       # 100KB total for CSS
        
        template_ok = total_template_size <= max_template_size
        css_ok = total_css_size <= max_css_size
        
        if not template_ok or not css_ok:
            print(f"❌ File sizes too large (templates: {total_template_size/1024:.1f}KB, CSS: {total_css_size/1024:.1f}KB)")
            return False
        
        print(f"✅ Template performance acceptable (templates: {total_template_size/1024:.1f}KB, CSS: {total_css_size/1024:.1f}KB)")
        return True
        
    except Exception as e:
        print(f"❌ Template performance test failed: {e}")
        return False


def run_template_tests():
    """Run all template rendering tests"""
    print("\n🧪 Template Rendering Test Suite")
    print("=" * 45)
    
    test_functions = [
        test_template_files_exist,
        test_base_template_structure,
        test_dashboard_template_structure,
        test_css_files_exist,
        test_css_theme_structure,
        test_agent_cards_css,
        test_javascript_functionality,
        test_accessibility_features,
        test_template_performance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 45)
    print(f"🏁 Template Rendering Test Results")
    print("=" * 45)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All template rendering tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_template_tests()
    sys.exit(0 if success else 1)