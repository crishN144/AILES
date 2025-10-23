# Keep Mac Awake for HPC Jobs - Complete Options

## 📊 Comparison of Methods

| Method | Reliability | Mac Can Sleep? | Mac Can Disconnect? | Recommended |
|--------|-------------|----------------|---------------------|-------------|
| **System Settings Only** | ⚠️ Medium | ❌ No | ❌ No | ❌ Not recommended |
| **caffeinate** | ✅ Good | ❌ No | ❌ No | ⚠️ OK for short jobs |
| **screen/tmux** | ✅✅ Excellent | ✅ Yes | ✅ Yes | ✅✅ **BEST** |

---

## Option 1: Mac System Settings ⚠️

**Pros:**
- Simple, just click toggles
- No commands needed

**Cons:**
- ❌ Mac must stay on
- ❌ WiFi must stay connected
- ❌ If Mac sleeps, SSH disconnects → job STOPS
- ❌ Can't close lid
- ❌ Battery drain overnight

**Use for:** Short jobs (<2 hours)

---

## Option 2: `caffeinate` Command ✅

**Setup:**
```bash
# On your Mac, before SSH:
caffeinate -i

# Keep this terminal open, use another terminal for SSH
```

Or:
```bash
# Wrap your SSH command:
caffeinate -i ssh bgxp240@hpc.university.edu
```

**Pros:**
- ✅ Prevents Mac sleep
- ✅ Simple command
- ✅ Built into macOS

**Cons:**
- ❌ Mac must stay on
- ❌ WiFi must stay connected
- ❌ If terminal closes, caffeinate stops
- ❌ Can't close Mac lid
- ❌ Battery drain

**Use for:** Medium jobs (2-8 hours)

---

## Option 3: `screen` or `tmux` on HPC ✅✅ **RECOMMENDED**

**Setup:**
```bash
# 1. SSH to HPC
ssh bgxp240@hpc.university.edu

# 2. Start screen
screen -S ailes_dataset

# 3. Run your job
cd /users/bgxp240/ailes_legal_ai
python3 data/raw/prod_dataset_analyzer.py

# 4. Detach: Ctrl+A then D
```

**Pros:**
- ✅✅ **Job runs on HPC, not Mac**
- ✅✅ **Mac can sleep/close/disconnect**
- ✅✅ **WiFi can drop - job continues**
- ✅✅ **Can reconnect from anywhere**
- ✅✅ **Can check progress anytime**
- ✅✅ **Most reliable for long jobs**
- ✅✅ **Zero battery drain**

**Cons:**
- Need to learn 2 commands (screen -S, screen -r)

**Use for:** Long jobs (8+ hours) ⭐ **YOUR CASE**

---

## 🎯 What You Should Do for 15-Hour Job

### **RECOMMENDED APPROACH:**

**Use `screen` on HPC** (completely independent of Mac!)

```bash
# Step 1: Check if screen is installed on HPC
screen --version

# If not installed, use tmux instead:
tmux --version

# Step 2: Start screen session
screen -S ailes_dataset

# Step 3: Run job
cd /users/bgxp240/ailes_legal_ai
python3 data/raw/prod_dataset_analyzer.py

# Step 4: Let it run for 1 minute to verify, then detach
# Press: Ctrl+A, release, then press D

# Step 5: You'll see:
# [detached from 12345.ailes_dataset]

# Step 6: Close Mac, go to sleep! Job keeps running on HPC
exit  # (exit SSH)

# Step 7: Next morning, check progress
ssh bgxp240@hpc.university.edu
screen -r ailes_dataset
```

---

## 🔧 Backup Plan (If screen/tmux not available)

If HPC doesn't have screen/tmux:

### Use `nohup` (No Hangup)

```bash
# SSH to HPC
ssh bgxp240@hpc.university.edu

# Run with nohup
cd /users/bgxp240/ailes_legal_ai
nohup python3 data/raw/prod_dataset_analyzer.py > output.log 2>&1 &

# Get process ID
echo $!

# Detach and close SSH
exit

# Check progress later
ssh bgxp240@hpc.university.edu
tail -f /users/bgxp240/ailes_legal_ai/ailes_pipeline_production.log
```

**Then on Mac:** Just keep it plugged in, settings you configured are fine!

---

## ⚡ Quick Decision Tree

**Q: Is job running on HPC (remote server)?**
- YES → Use `screen` or `tmux` ✅✅ (Mac can sleep)
- NO → Job on Mac? Read below ↓

**Q: Job running locally on Mac?**
- YES → Use `caffeinate -i` + keep Mac awake
- NO → You're using HPC! Use screen!

**Your case:** Job runs on HPC → **Use screen!** ⭐

---

## 📱 Bonus: Check Progress from Phone

With screen, you can even check progress from your phone!

1. Install SSH app (Termius, Prompt, etc.)
2. SSH to HPC from phone
3. Run: `tail -20 /users/bgxp240/ailes_legal_ai/ailes_pipeline_production.log`
4. Or reattach: `screen -r ailes_dataset`

---

## Summary

**For your 15-hour dataset generation:**

✅ **Use `screen` on HPC** (BEST)
- Mac can sleep
- WiFi can disconnect  
- Job continues regardless
- Zero Mac battery drain

✅ **Mac settings as backup**
- Keep "Prevent sleep on power adapter" ON
- Just in case you need to run something locally

✅ **Result:** Sleep peacefully, wake up to 15K pairs! 😴→☕→🎉

