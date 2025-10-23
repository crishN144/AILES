# Using GNU Screen for Long-Running Jobs

## Why Use Screen?
- Job continues even if SSH disconnects
- Mac can sleep/close - job keeps running on HPC
- Can reconnect from anywhere
- Most reliable for 15+ hour jobs

## Quick Start Guide

### 1. Start a Screen Session
```bash
# SSH to HPC first
ssh bgxp240@hpc.youruniversity.edu

# Start named screen session
screen -S ailes_dataset

# You're now inside screen! Run your job:
cd /users/bgxp240/ailes_legal_ai
python3 data/raw/prod_dataset_analyzer.py
```

### 2. Detach from Screen (Keep Job Running)
Press: `Ctrl+A` then `D` (detach)

Your job keeps running! You can now:
- Close your Mac
- Disconnect WiFi
- Go to sleep
- The job continues on HPC!

### 3. Reconnect Later
```bash
# SSH back to HPC
ssh bgxp240@hpc.youruniversity.edu

# List screen sessions
screen -ls

# Reattach to your session
screen -r ailes_dataset
```

You'll see exactly where you left off!

### 4. Check Progress While Detached
```bash
# Without entering screen, check progress:
tail -f /users/bgxp240/ailes_legal_ai/ailes_pipeline_production.log

# Check pairs count
wc -l /users/bgxp240/ailes_legal_ai/ailes_training_dataset_production.jsonl
```

## Common Screen Commands

| Action | Command |
|--------|---------|
| Create session | `screen -S name` |
| Detach | `Ctrl+A` then `D` |
| List sessions | `screen -ls` |
| Reattach | `screen -r name` |
| Kill session | `screen -X -S name quit` |
| Scroll up | `Ctrl+A` then `Esc`, then arrow keys |

## Full Example Workflow

```bash
# 1. SSH to HPC
ssh bgxp240@hpc.youruniversity.edu

# 2. Start screen
screen -S ailes_dataset

# 3. Run job
cd /users/bgxp240/ailes_legal_ai
python3 data/raw/prod_dataset_analyzer.py

# 4. Wait a minute to see it working, then detach
# Press: Ctrl+A, then D

# 5. Close Mac, go to bed, disconnect - job keeps running!

# 6. Next morning, reconnect
ssh bgxp240@hpc.youruniversity.edu
screen -r ailes_dataset

# 7. See progress!
```

## Advantages Over Caffeinate

✅ Job runs on HPC (not dependent on your Mac)
✅ Can disconnect and reconnect anytime
✅ Mac can sleep/restart - job unaffected
✅ Can check progress without interrupting
✅ Most reliable for multi-day jobs

## If Job Completes While Detached

Screen session will stay alive. When you reattach:
```bash
screen -r ailes_dataset
```

You'll see:
```
Generated 15000 quality pairs for [final file].xml in 45.2s
Files processed: 1829
```

Then you can safely exit screen and kill it:
```bash
# Inside screen: Ctrl+D to exit
# Or from outside: screen -X -S ailes_dataset quit
```

---
**Pro Tip:** You can even check progress from your phone using a mobile SSH app!
