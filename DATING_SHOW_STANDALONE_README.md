# Dating Show Standalone Simulation

This setup allows you to run the dating show simulation using only `@dating_show/` and `@dating_show_env/` directories, without requiring the separate `@reverie/` directory.

## What Was Done

1. **Copied reverie components** into `dating_show/reverie_core/`:
   - `reverie.py` - Main ReverieServer class
   - `global_methods.py` - Utility functions (updated paths)
   - `maze.py` - Spatial navigation system
   - `persona/` - Agent logic and behavior
   - `utils.py` - Additional utilities

2. **Updated paths** to use `dating_show_env/frontend_service/` instead of `environment/frontend_server/`

3. **Created storage directories** in `dating_show_env/frontend_service/`:
   - `storage/` - Contains simulation data
   - `temp_storage/` - For frontend communication

4. **Copied base simulation** from original storage to new location

## Files Structure

```
dating_show/
├── reverie_core/           # Local copy of reverie components
│   ├── reverie.py         # Main simulation server
│   ├── global_methods.py  # Updated with new paths
│   ├── maze.py           # Spatial navigation
│   ├── persona/          # Agent behavior logic
│   └── utils.py          # Utilities
├── standalone_simulation.py  # New standalone runner
└── [other dating_show files]

dating_show_env/
└── frontend_service/
    ├── storage/           # Simulation data storage
    │   └── base_the_ville_n25/  # Base 25-agent simulation
    ├── temp_storage/      # Frontend communication files
    └── static_dirs/       # Assets and matrix data
        └── assets/
            └── the_ville/
                └── matrix/    # Maze configuration files

run_dating_show_standalone.py  # Main entry point
```

## How to Run

### Method 1: Direct standalone runner
```bash
python run_dating_show_standalone.py
```

### Method 2: From dating_show directory
```bash
cd dating_show
python standalone_simulation.py
```

### Command line options:
- `--auto 10` - Run 10 simulation steps automatically
- `--step` - Run single step
- No arguments - Interactive mode

## Interactive Commands

When running in interactive mode:
- `step` - Run one simulation step
- `auto N` - Auto-run N steps (default: 10)
- `status` - Show current simulation status
- `save` - Save simulation state
- `quit` - Exit simulation

## Dependencies

The simulation should work with:
- Python 3.7+
- Required packages from dating_show dependencies
- Matrix data in `dating_show_env/frontend_service/static_dirs/assets/the_ville/matrix/`

## Frontend Integration

The simulation creates frontend sync files in:
- `dating_show_env/frontend_service/temp_storage/curr_sim_code.json`
- `dating_show_env/frontend_service/temp_storage/curr_step.json`

These allow the frontend to track simulation progress.

## Troubleshooting

1. **Import errors**: Ensure all files are copied correctly to `reverie_core/`
2. **Path errors**: Check that `dating_show_env/frontend_service/` structure exists
3. **Matrix errors**: Verify matrix files exist in `static_dirs/assets/the_ville/matrix/`
4. **Storage errors**: Ensure base simulation exists in `storage/base_the_ville_n25/`

## Benefits

- **Self-contained**: Only requires `@dating_show/` and `@dating_show_env/`
- **No external dependencies**: Doesn't rely on separate `@reverie/` directory
- **Portable**: Can be moved/deployed as a unit
- **Isolated**: Changes don't affect original reverie implementation