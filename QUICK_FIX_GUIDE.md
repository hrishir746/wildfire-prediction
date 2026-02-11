# Quick Fix Guide: UnicodeEncodeError in Rovo Dev

**Problem:** Getting `UnicodeEncodeError: 'charmap' codec can't encode characters...`

**Solution:** Use the safe wrapper script!

## ✅ Immediate Fix (30 seconds)

Instead of running:
```bash
python scripts/train_extended_gpu.py
```

Run:
```bash
python scripts/run_training_safe.py train_extended_gpu
```

That's it! The wrapper handles all UTF-8 encoding issues automatically.

## 📋 Quick Reference

### Training Commands

| Old Command | New Command (Safe) |
|-------------|-------------------|
| `python scripts/train_extended_gpu.py` | `python scripts/run_training_safe.py train_extended_gpu` |
| `python scripts/train_with_dashboard.py --epochs 50` | `python scripts/run_training_safe.py train_with_dashboard --epochs 50` |
| `python scripts/train_model.py --epochs 20` | `python scripts/run_training_safe.py train_model --epochs 20` |

### Validation Commands

| Old Command | New Command (Safe) |
|-------------|-------------------|
| `python scripts/validate_extended_model.py` | `python scripts/run_training_safe.py validate_extended_model` |
| `python scripts/validate_80_percent.py` | `python scripts/run_training_safe.py validate_80_percent` |
| `python scripts/validate_camp_2018.py` | `python scripts/run_training_safe.py validate_camp_2018` |

### Evaluation Commands

| Old Command | New Command (Safe) |
|-------------|-------------------|
| `python scripts/evaluate_model.py` | `python scripts/run_training_safe.py evaluate_model` |
| `python scripts/evaluate_real_fires.py --event camp_2018` | `python scripts/run_training_safe.py evaluate_real_fires --event camp_2018` |

### Test Commands

| Old Command | New Command (Safe) |
|-------------|-------------------|
| `python scripts/test_full_pipeline_512.py` | `python scripts/run_training_safe.py test_full_pipeline_512` |
| `python scripts/quick_demo.py` | `python scripts/run_training_safe.py quick_demo` |

## 🔍 How It Works

The wrapper script:
1. Forces UTF-8 encoding **before** any other code runs
2. Sets multiple environment variables (`PYTHONIOENCODING`, `PYTHONUTF8`)
3. Reconfigures stdout/stderr with error handling
4. Then runs your target script normally

## 📚 Available Scripts

Run `python scripts/run_training_safe.py` to see all available scripts:

**Training:**
- train_extended_gpu
- train_gpu_optimized
- train_fast_production
- train_model
- train_with_dashboard
- finetune_gpu

**Validation:**
- validate_80_percent
- validate_extended_model
- validate_camp_2018
- wildfire_validation

**Evaluation:**
- evaluate_real_fires
- evaluate_model
- quick_eval_gpu

**Demo:**
- comprehensive_demo
- quick_demo

**Test:**
- test_full_pipeline_512

## 💡 Why This Happens

The Rovo Dev extension spawns child processes that don't inherit UTF-8 encoding from the parent process. They default to Windows-1252 (charmap) encoding, which can't handle special characters like ✓, °, →, etc.

## 🛠️ What We Did

1. **Created safe wrapper** (`scripts/run_training_safe.py`)
   - Multi-level UTF-8 enforcement
   - Works with Rovo Dev child processes

2. **Replaced Unicode characters** (18 scripts)
   - ✓ → [OK]
   - ° → deg
   - → → ->
   - etc.

3. **Tested successfully**
   - No encoding errors
   - All scripts work normally

## 🔄 Alternative: Direct Execution

Since we also replaced all Unicode characters with ASCII equivalents, you can still run scripts directly:

```bash
python scripts/train_extended_gpu.py
```

But the wrapper provides extra safety and is recommended for Rovo Dev users.

## ❓ Troubleshooting

**Q: Still getting Unicode errors?**  
A: Make sure you're using the wrapper: `python scripts/run_training_safe.py SCRIPT_NAME`

**Q: Script not found in wrapper?**  
A: Check the available scripts list with: `python scripts/run_training_safe.py`

**Q: Want to restore original Unicode characters?**  
A: Backups are in `scripts/*.unicode_backup` - just rename them back to `.py`

**Q: Need to run in WSL2 for GPU?**  
A: Yes! Use: `wsl python3 scripts/run_training_safe.py train_extended_gpu`

## 📖 Full Documentation

For complete details, see:
- **UNICODE_FIX_SOLUTION.md** - Comprehensive solution documentation
- **GPU_TRAINING_NEXT_STEPS.md** - GPU training setup and commands
- **UTF8_AND_512_UPGRADE_SUMMARY.md** - All recent changes

---

**Status:** ✅ Fixed  
**Time to fix:** 30 seconds (just use the wrapper)  
**Compatibility:** Windows, Linux, macOS  
**Tested:** Successfully with Rovo Dev extension on Windows 11
