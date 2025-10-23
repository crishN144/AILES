# AILES Dataset Generation - Quick Reference

## Current Status (as of Oct 23, 01:06 AM)
- **Progress:** 4,743 / 15,000 pairs (31.6%)
- **Quality:** 8/10 - Production ready
- **Completion:** Thursday, Oct 23 at 4:17 PM

## Important Files
```
Dataset:     /users/bgxp240/ailes_legal_ai/ailes_training_dataset_production.jsonl
Generator:   /users/bgxp240/ailes_legal_ai/data/raw/prod_dataset_analyzer.py
Checkpoint:  /users/bgxp240/ailes_legal_ai/ailes_checkpoints/tracker_state.pkl
Log:         /users/bgxp240/ailes_legal_ai/ailes_pipeline_production.log
```

## If Generation Stops/Disconnects

### ✅ YOU ARE SAFE - Checkpoint system active!

**To resume:**
```bash
cd /users/bgxp240/ailes_legal_ai
python3 data/raw/prod_dataset_analyzer.py
```

Will automatically:
- Load checkpoint (580+ files already done)
- Skip processed files
- Continue where it left off
- Append to existing dataset

## To Increase Target to 25K (Recommended)

1. Stop current process: `Ctrl+C`
2. Edit: `nano data/raw/prod_dataset_analyzer.py`
3. Line 50: Change `TARGET_PAIRS = 15000` → `TARGET_PAIRS = 25000`
4. Save and restart: `python3 data/raw/prod_dataset_analyzer.py`

## Check Progress

```bash
# Current pairs
wc -l ailes_training_dataset_production.jsonl

# Latest log entries
tail -20 ailes_pipeline_production.log

# Processing rate
grep "quality pairs" ailes_pipeline_production.log | tail -10
```

## Quality Metrics

| Metric | Score | Grade |
|--------|-------|-------|
| Factuality | 100% | A+ |
| Template pollution | 0% | A+ |
| Question diversity | 99% | A+ |
| Statutory accuracy | 76% | C+ |
| Overall | 8/10 | B+ |

## Troubleshooting

**If process crashes:**
```bash
# Check if running
ps aux | grep prod_dataset_analyzer

# Check last error
tail -50 ailes_pipeline_production.log

# Restart with checkpoint
python3 data/raw/prod_dataset_analyzer.py
```

**If out of space:**
```bash
df -h /users/bgxp240
# You have 97 TB available - not an issue
```

**If need to check checkpoint:**
```bash
ls -lh ailes_checkpoints/tracker_state.pkl
```

## Next Steps After Completion

1. **Verify dataset:**
   ```bash
   wc -l ailes_training_dataset_production.jsonl
   # Should be ~15,000 (or 25,000 if changed)
   ```

2. **Start fine-tuning:**
   - Model: meta-llama/Llama-3.1-8B
   - Method: LoRA fine-tuning
   - Time: 18-24 hours
   - VRAM: 16 GB

3. **Implement RAG (optional):**
   - Guide: `/users/bgxp240/ailes_legal_ai/RAG_ARCHITECTURE_GUIDE.md`
   - Vector DB: Chroma
   - Embeddings: bge-large-en-v1.5

---
Last updated: Oct 23, 2025 01:36 AM
