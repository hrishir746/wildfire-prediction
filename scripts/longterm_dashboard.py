#!/usr/bin/env python3
"""
Real-time dashboard for autonomous training pipeline.
Shows: Dataset generation → Training → MTBS validation
"""
import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Autonomous Training Pipeline", page_icon="🔥", layout="wide")

st.title("🔥 Autonomous Wildfire Training Pipeline")
st.markdown("**Goal:** Achieve 30-60% IoU on MTBS Camp Fire 2018 validation")

# Alert system
alert_file = ROOT / "outputs" / "pipeline_alerts.json"
if alert_file.exists():
    with open(alert_file) as f:
        alerts = json.load(f)
    if alerts:
        for alert in alerts[-3:]:  # Show last 3 alerts
            if alert['type'] == 'error':
                st.error(f"🚨 {alert['time']}: {alert['message']}")
            elif alert['type'] == 'warning':
                st.warning(f"⚠️ {alert['time']}: {alert['message']}")
            elif alert['type'] == 'success':
                st.success(f"✅ {alert['time']}: {alert['message']}")

# Check pipeline status
longterm_dataset = ROOT / "data" / "longterm_fires" / "longterm_dataset.npz"
longterm_model = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"
metrics_file = ROOT / "outputs" / "longterm_training_metrics.json"

col1, col2, col3 = st.columns(3)

# Stage 1: Dataset Generation
with col1:
    st.subheader("📊 Stage 1: Dataset")
    if longterm_dataset.exists():
        file_size = longterm_dataset.stat().st_size / 1e6
        st.success(f"✅ Complete ({file_size:.1f} MB)")
        try:
            data = np.load(longterm_dataset)
            st.metric("Samples", len(data['X']))
            st.metric("Timesteps", "150 (vs 10 before)")
        except Exception as e:
            st.info("Dataset file in use by training")
            st.metric("Size", f"{file_size:.1f} MB")
    else:
        # Check for progress file
        progress_file = ROOT / "data" / "longterm_fires" / "generation_progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            st.warning(f"⏳ Generating... {progress['completed']}/{progress['total']}")
            st.progress(progress['percentage'] / 100)
            st.caption(f"{progress['percentage']:.1f}% complete")
            
            # ETA calculation
            if progress['completed'] > 5:
                samples_per_min = progress['completed'] / ((datetime.now() - datetime(2026, 2, 7, 19, 45)).total_seconds() / 60)
                remaining_samples = progress['total'] - progress['completed']
                eta_min = remaining_samples / samples_per_min
                st.info(f"⏱️ ETA: ~{int(eta_min)} minutes")
        else:
            st.warning("⏳ Starting...")
            st.info("200 samples × 150 timesteps")

# Stage 2: Training
with col2:
    st.subheader("🔥 Stage 2: Training")
    
    # Check for live training progress
    training_progress_file = ROOT / "outputs" / "training_live_progress.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        st.success("✅ Complete")
        st.metric("Best Val IoU", f"{metrics['best_val_iou']:.4f}")
        st.metric("Training Time", f"{metrics['training_time_minutes']:.1f} min")
        
        # Show training curves
        if 'history' in metrics:
            import pandas as pd
            import plotly.express as px
            
            history = metrics['history']
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            df = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss': history['train_loss'],
                'Val Loss': history['val_loss'],
                'Val IoU': history['val_iou'],
            })
            
            fig = px.line(df, x='Epoch', y=['Train Loss', 'Val Loss'], 
                         title='Training Progress')
            st.plotly_chart(fig, use_container_width=True)
            
    elif training_progress_file.exists():
        # Show LIVE training progress
        with open(training_progress_file) as f:
            progress = json.load(f)
        
        st.warning(f"⏳ Training... Epoch {progress['current_epoch']}/{progress['total_epochs']}")
        st.progress(progress['current_epoch'] / progress['total_epochs'])
        
        if 'train_loss' in progress:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Train Loss", f"{progress['train_loss']:.4f}")
            with col_b:
                st.metric("Val IoU", f"{progress.get('val_iou', 0):.4f}")
        
        if progress['current_epoch'] > 1:
            eta_min = progress.get('eta_minutes', 0)
            st.info(f"⏱️ ETA: ~{int(eta_min)} minutes")
            
    elif longterm_model.exists():
        st.info("✅ Model saved")
    elif longterm_dataset.exists():
        st.warning("⏳ Ready to start")
        st.info("40 epochs, ~2-3 hours")
    else:
        st.info("⏸️ Waiting for dataset")

# Stage 3: MTBS Validation
with col3:
    st.subheader("🎯 Stage 3: Validation")
    
    mtbs_results = ROOT / "outputs" / "LONGTERM_MTBS_RESULTS.txt"
    if mtbs_results.exists():
        with open(mtbs_results) as f:
            content = f.read()
        st.success("✅ Validated on MTBS")
        
        # Parse IoU from results
        for line in content.split('\n'):
            if 'IoU:' in line:
                st.metric("MTBS Camp Fire IoU", line.split(':')[1].strip())
                break
        
        st.text_area("Results", content, height=200)
    elif longterm_model.exists():
        st.warning("⏳ Ready to validate")
        st.info("Will test on Camp Fire 2018")
        st.info("Target: 30-60% IoU")
    else:
        st.info("⏸️ Waiting for training")

# Timeline
st.markdown("---")
st.subheader("📅 Pipeline Timeline")

timeline_data = []
if longterm_dataset.exists():
    timeline_data.append(f"✅ Dataset generated: {datetime.fromtimestamp(longterm_dataset.stat().st_mtime).strftime('%H:%M')}")
if longterm_model.exists():
    timeline_data.append(f"✅ Training complete: {datetime.fromtimestamp(longterm_model.stat().st_mtime).strftime('%H:%M')}")
if mtbs_results.exists():
    timeline_data.append(f"✅ MTBS validation: {datetime.fromtimestamp(mtbs_results.stat().st_mtime).strftime('%H:%M')}")

if timeline_data:
    for item in timeline_data:
        st.markdown(item)
else:
    st.info("Pipeline starting...")

# Current status message & countdown timer
st.markdown("---")
current_time = datetime.now().strftime("%H:%M:%S")

# Calculate next check-in time (20 min intervals)
import math
minutes_since_start = (datetime.now() - datetime(2026, 2, 7, 19, 45)).total_seconds() / 60
minutes_to_next = 20 - (minutes_since_start % 20)
next_checkin = datetime.now().timestamp() + (minutes_to_next * 60)
next_checkin_str = datetime.fromtimestamp(next_checkin).strftime("%H:%M")

col_status, col_timer = st.columns([2, 1])

with col_status:
    if not longterm_dataset.exists():
        st.info(f"🔄 [{current_time}] Generating long-term dataset...")
    elif not longterm_model.exists():
        st.info(f"🔄 [{current_time}] Dataset ready. Training will start soon...")
    elif not mtbs_results.exists():
        st.info(f"🔄 [{current_time}] Training complete. Validation pending...")
    else:
        st.success(f"✅ [{current_time}] Pipeline complete! Check MTBS results above.")

with col_timer:
    st.metric("⏰ Next Check-in", next_checkin_str)
    st.caption(f"In {int(minutes_to_next)} minutes")

# Auto-refresh every 5 seconds for live updates
time.sleep(5)
st.rerun()
