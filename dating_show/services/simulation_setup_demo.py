#!/usr/bin/env python3
"""
Demonstration script for the SimulationSetupService

This script shows how to use the automated 8-agent simulation setup service
for creating, validating, and managing dating show simulations.

Usage:
    python simulation_setup_demo.py [command]
    
Commands:
    status    - Show current simulation status
    validate  - Validate simulation consistency
    create    - Create new 8-agent simulation (with confirmation)
    repair    - Repair any issues with existing simulation
    cleanup   - Remove extra personas if any exist
    
If no command is provided, shows status by default.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path so we can import the service
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dating_show.services.simulation_setup_service import (
    SimulationSetupService,
    create_dating_show_simulation,
    validate_dating_show_simulation,
    repair_dating_show_simulation,
    get_dating_show_status
)


def print_status(status_data):
    """Print simulation status in a readable format"""
    print("=== Simulation Status ===")
    print(f"📁 Exists: {'✅ Yes' if status_data['exists'] else '❌ No'}")
    print(f"🔍 Valid: {'✅ Yes' if status_data['is_valid'] else '❌ No'}")
    print(f"👥 Agents: {status_data['agent_count']}/8")
    
    if status_data['exists']:
        print(f"📅 Last Modified: {status_data['last_modified']}")
        
        if status_data['missing_agents']:
            print(f"⚠️  Missing Agents: {', '.join(status_data['missing_agents'])}")
            
        if status_data['extra_agents']:
            print(f"🚨 Extra Agents: {', '.join(status_data['extra_agents'])}")
            
        print("\n📋 File Status:")
        for file_path, file_info in status_data['files_status'].items():
            status_icon = "✅" if file_info['exists'] else "❌"
            file_type = "📁" if file_info['is_dir'] else "📄"
            print(f"   {status_icon} {file_type} {file_path}")


def print_validation_results(validation_data):
    """Print validation results in a readable format"""
    print("=== Validation Results ===")
    print(f"🔍 Overall: {'✅ PASSED' if validation_data['is_valid'] else '❌ FAILED'}")
    print(f"🧪 Checks Performed: {len(validation_data['checks_performed'])}")
    
    for check in validation_data['checks_performed']:
        print(f"   ✓ {check}")
        
    if validation_data['errors']:
        print(f"\n🚨 Errors ({len(validation_data['errors'])}):")
        for error in validation_data['errors']:
            print(f"   • {error}")
            
    if validation_data['warnings']:
        print(f"\n⚠️  Warnings ({len(validation_data['warnings'])}):")
        for warning in validation_data['warnings']:
            print(f"   • {warning}")


def print_operation_results(result_data, operation_name):
    """Print operation results in a readable format"""
    print(f"=== {operation_name} Results ===")
    print(f"🎯 Success: {'✅ Yes' if result_data['success'] else '❌ No'}")
    print(f"💬 Message: {result_data['message']}")
    
    if 'agents_created' in result_data and result_data['agents_created']:
        print(f"👥 Agents Created: {len(result_data['agents_created'])}")
        for agent in result_data['agents_created']:
            print(f"   • {agent}")
            
    if 'files_processed' in result_data and result_data['files_processed']:
        print(f"📄 Files Processed: {len(result_data['files_processed'])}")
        for file_path in result_data['files_processed']:
            print(f"   • {file_path}")
            
    if 'removed_personas' in result_data and result_data['removed_personas']:
        print(f"🗑️  Removed Personas: {len(result_data['removed_personas'])}")
        for persona in result_data['removed_personas']:
            print(f"   • {persona}")
            
    if 'repairs_attempted' in result_data and result_data['repairs_attempted']:
        print(f"🔧 Repairs Attempted: {len(result_data['repairs_attempted'])}")
        for repair in result_data['repairs_attempted']:
            status = "✅" if repair in result_data.get('repairs_successful', []) else "❌"
            print(f"   {status} {repair}")
            
    if result_data.get('errors'):
        print(f"\n🚨 Errors ({len(result_data['errors'])}):")
        for error in result_data['errors']:
            print(f"   • {error}")


def cmd_status(args):
    """Handle status command"""
    print("Getting simulation status...\n")
    status = get_dating_show_status()
    print_status(status)


def cmd_validate(args):
    """Handle validate command"""
    print("Validating simulation...\n")
    validation = validate_dating_show_simulation()
    print_validation_results(validation)


def cmd_create(args):
    """Handle create command"""
    # Check if simulation already exists
    status = get_dating_show_status()
    
    if status['exists'] and not args.force:
        print("⚠️  Simulation already exists!")
        print_status(status)
        print("\nUse --force to recreate the simulation.")
        return
        
    if status['exists'] and args.force:
        print("🔄 Recreating existing simulation...\n")
    else:
        print("🆕 Creating new simulation...\n")
        
    result = create_dating_show_simulation(force_recreate=args.force)
    print_operation_results(result, "Creation")
    
    if result['success']:
        print("\n🎉 Running post-creation validation...")
        validation = validate_dating_show_simulation()
        print_validation_results(validation)


def cmd_repair(args):
    """Handle repair command"""
    print("Attempting to repair simulation...\n")
    result = repair_dating_show_simulation()
    print_operation_results(result, "Repair")


def cmd_cleanup(args):
    """Handle cleanup command"""
    print("Cleaning up extra personas...\n")
    service = SimulationSetupService()
    result = service.cleanup_extra_personas()
    print_operation_results(result, "Cleanup")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Automated 8-Agent Dating Show Simulation Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simulation_setup_demo.py status
    python simulation_setup_demo.py validate  
    python simulation_setup_demo.py create --force
    python simulation_setup_demo.py repair
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show simulation status')
    
    # Validate command  
    subparsers.add_parser('validate', help='Validate simulation consistency')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create 8-agent simulation')
    create_parser.add_argument('--force', action='store_true', 
                              help='Force recreation if simulation exists')
    
    # Repair command
    subparsers.add_parser('repair', help='Repair simulation issues')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Remove extra personas')
    
    args = parser.parse_args()
    
    # Default to status if no command provided
    if not args.command:
        args.command = 'status'
    
    # Route to appropriate command handler
    command_handlers = {
        'status': cmd_status,
        'validate': cmd_validate, 
        'create': cmd_create,
        'repair': cmd_repair,
        'cleanup': cmd_cleanup
    }
    
    try:
        handler = command_handlers[args.command]
        handler(args)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n🚨 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()