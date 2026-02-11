# Bridge Domain Gap - Path B: Realistic Synthetic Data

## Current Status
- ✅ Synthetic training: 91-93% accuracy
- ❌ Real Camp Fire: 0% accuracy
- ⚠️ Domain gap identified and quantified

## Solution: Generate California-Realistic Synthetic Data

### Phase 1: Data Generation (4-6 hours)
**Goal:** Create 1000 synthetic samples matching real CA wildfire terrain

1. **Sample real CA terrain statistics** (1 hour)
   - Extract slope/aspect/elevation distributions from USGS 3DEP
   - Sample Butte County, Sonoma, Santa Barbara regions
   - Get LANDFIRE fuel type distributions
   - Characterize typical CA fire conditions (wind, weather)

2. **Update synthetic generator** (2 hours)
   - Modify `rothermel_simulator_gpu.py` terrain generation
   - Match real slope distributions (CA: 5-35° common)
   - Match real aspect patterns (coastal vs inland)
   - Use real fuel density distributions from LANDFIRE
   - Use realistic wind patterns (15-40 mph, westerly)

3. **Generate realistic dataset** (2-3 hours)
   - 1000 samples at 512x512
   - GPU-accelerated generation (we have this!)
   - Save with metadata
   - Expected time: ~2-3 hours with GPU

### Phase 2: Model Training (4-6 hours)
**Goal:** Train model on realistic data

1. **Train with GPU** (3-4 hours)
   ```bash
   python scripts/train_with_dashboard_gpu.py \
       --epochs 50 \
       --samples 1000 \
       --grid-size 512 \
       --batch-size 4
   ```

2. **Monitor training** (ongoing)
   - Watch for convergence
   - Validate on held-out realistic synthetic data
   - Target: >85% on synthetic

### Phase 3: Validation on Real Fires (2-3 hours)
**Goal:** Test on Camp Fire 2018 and other real fires

1. **Validate on Camp Fire** (30 min)
   ```bash
   python scripts/validate_camp_2018.py
   ```
   - Expected: 60-80% accuracy (realistic terrain match)

2. **Test on additional fires** (1 hour)
   - Use other CA fires from MTBS
   - Show generalization

3. **Create validation report** (1 hour)
   - Document results
   - Show improvement from 0% → 70%+
   - Explain remaining gap

### Phase 4: Documentation (1-2 hours)
**Goal:** Professional documentation

1. **Results summary**
   - Before: 0% on real fires
   - After: 70-80% on real fires
   - Methodology: Domain-adapted synthetic data

2. **Visualizations**
   - Side-by-side predictions vs actual
   - Training curves
   - Error analysis

## Timeline
- **Day 1 (8-10 hours)**
  - Morning: Phase 1 (terrain statistics + generator update)
  - Afternoon: Phase 2 start (begin training)
  - Evening: Monitor training

- **Day 2 (6-8 hours)**
  - Morning: Phase 2 complete + Phase 3 (validation)
  - Afternoon: Phase 4 (documentation)
  - Evening: Final testing and polish

## Expected Outcomes
- ✅ 70-80% accuracy on real Camp Fire data
- ✅ Generalizable model (not overfitted to specific fires)
- ✅ Documented methodology
- ✅ Production-ready for California wildfires

## Alternative: Path C (2 hours, safer)
If you run into time issues or want a guaranteed submission:
- Document the domain gap (scientifically honest)
- Show 91% synthetic, 0% real
- Propose Path B as "future work"
- This is still a strong submission showing ML understanding

## My Recommendation
**Start with Path B** - you have the GPU infrastructure and skills. If you hit a blocker after Day 1, pivot to Path C for documentation. You'll still have learned a lot and have something to submit.

---
**Ready to start?** I can help with each phase!
