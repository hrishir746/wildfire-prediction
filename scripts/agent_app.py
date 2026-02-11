#!/usr/bin/env python3
"""
Interactive conversational interface for wildfire prediction.

Educational interface: users ask questions about fire spread predictions,
model confidence, environmental factors. Read-only, interpretive agent.

Two modes:
1. CLI (interactive terminal): python scripts/agent_app.py --cli --checkpoint ...
2. Streamlit (web UI): streamlit run scripts/agent_app.py

Example:
  # CLI mode
  python scripts/agent_app.py --cli --checkpoint outputs/checkpoints/unet_final.pt --input outputs/real_data/real_input_8ch.npy
  
  # Streamlit mode
  streamlit run scripts/agent_app.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# Try Streamlit first; fall back to CLI
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.agent import FirePredictionAgent


def run_cli_agent():
    """Interactive CLI mode."""
    parser = argparse.ArgumentParser(description="Interactive wildfire prediction agent (CLI)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", type=Path, help="Path to input .npy file (8-channel)")
    parser.add_argument("--baseline", type=Path, help="Path to baseline prediction .npy (for comparison)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing agent (device: {device})...")
    agent = FirePredictionAgent(args.checkpoint, device=device)
    print("Agent ready. Type 'help' for options, 'quit' to exit.\n")

    # Load and run initial prediction if input provided
    if args.input:
        if not args.input.exists():
            print(f"Input file not found: {args.input}")
            return
        print(f"Loading input from {args.input}...")
        input_stack = np.load(args.input)
        print(f"Input shape: {input_stack.shape}")
        print("Running prediction...")
        pred = agent.predict_fire_spread(input_stack)
        print(f"Prediction complete. Mean probability: {pred['confidence']:.3f}")
        print()

    baseline = None
    if args.baseline:
        if args.baseline.exists():
            baseline = np.load(args.baseline)
            print(f"Baseline loaded: {baseline.shape}\n")

    # Interactive loop
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if question.lower() == "help":
            print(
                "\nQuestions I can answer:\n"
                "- 'What is the predicted fire extent?'\n"
                "- 'Which areas are most at risk?'\n"
                "- 'How confident is the model?'\n"
                "- 'What environmental factors drive this?'\n"
                "- 'How does this compare to the baseline?'\n"
                "- 'What are the limitations?'\n"
                "- 'help' - show this message\n"
                "- 'quit' - exit\n"
            )
            continue

        if question.lower() == "status":
            if agent.last_prediction:
                pred = agent.last_prediction
                print(f"\nCurrent prediction:")
                print(f"  Confidence: {pred['confidence']:.3f}")
                print(f"  High-risk pixels (>70%): {pred['high_confidence_pixels']}")
                print(f"  Affected area: {(pred['extent'].mean()*100):.1f}%")
            else:
                print("No prediction loaded.")
            continue

        # Baseline comparison
        if "baseline" in question.lower() or "rothermel" in question.lower():
            if baseline is not None:
                answer = agent.compare_to_baseline(baseline)
            elif agent.last_prediction:
                answer = agent.compare_to_baseline(np.zeros_like(agent.last_prediction["extent"]))
            else:
                answer = "No prediction or baseline available."
        else:
            answer = agent.answer_question(question)

        print(f"\nAgent: {answer}\n")


def run_streamlit_app():
    """Streamlit web UI."""
    import matplotlib.pyplot as plt
    from src.data import WildfireSyntheticDataset
    from src.model import build_unet

    st.set_page_config(page_title="Wildfire Prediction -- Agent", layout="wide")

    st.title("Wildfire Spread Prediction System")
    st.caption("Educational interpretive agent. See SOURCES.md for scientific basis.")

    tab_overview, tab_agent, tab_limitations = st.tabs([
        "Overview", "Agent", "Limitations"
    ])

    with tab_overview:
        st.header("Project Overview")
        st.markdown("""
        This system predicts **short-term wildfire spread** using:
        - **Baseline:** Rothermel-inspired 2D simulator (Rothermel 1972; FlamMap).
        - **AI:** U-Net that takes 8 input channels (terrain, fuel, wind, initial fire) and outputs a **fire probability map** per pixel.

        **Pipeline:** Data -> Simulation -> Training -> Model -> Evaluation (IoU, Dice, RoS error).
        
        **Scientific Basis:** See SOURCES.md for full citations (Rothermel 1972, FlamMap, NASA, Sullivan 2009, Kim et al. 2023).
        """)

    with tab_agent:
        st.header("Wildfire Prediction Agent")
        
        ckpt_path = ROOT / "outputs" / "checkpoints" / "unet_final.pt"
        if not ckpt_path.exists():
            st.error("No trained model found. Run: `python scripts/train_model.py` first.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and sample
        @st.cache_resource
        def load_agent_and_data():
            agent = FirePredictionAgent(ckpt_path, device=device)
            ds = WildfireSyntheticDataset(num_samples=1, grid_size=(64, 64), horizon_steps=10, seed=77)
            x, y = ds[0]
            return agent, x, y

        agent, x, y = load_agent_and_data()

        # Run prediction on sample
        with torch.no_grad():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = ckpt.get("config", {})
            grid = config.get("simulation", {}).get("grid_size", [64, 64])
            grid_size = (int(grid[0]), int(grid[1]))
            in_ch = config.get("model", {}).get("input_channels", 8)
            out_ch = config.get("model", {}).get("output_channels", 1)
            model = build_unet(in_channels=in_ch, out_channels=out_ch)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            pred = torch.sigmoid(model(x.unsqueeze(0))).squeeze().numpy()
        target = y.squeeze().numpy()

        col1, col2 = st.columns(2)
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(target, cmap="hot", vmin=0, vmax=1)
            axes[0].set_title("Ground Truth (Rothermel Baseline)")
            axes[0].axis("off")
            axes[1].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("Model Prediction (Probability)")
            axes[1].axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Ask about the prediction:")
            question = st.text_input("Your question:")
            if question:
                agent.predict_fire_spread(x.numpy())
                answer = agent.answer_question(question)
                st.write(answer)

        st.subheader("Model Insights")
        if agent.last_prediction:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{agent.last_prediction['confidence']:.3f}")
            with col2:
                st.metric("High-Risk Pixels", agent.last_prediction['high_confidence_pixels'])
            with col3:
                st.metric("Affected Area %", f"{(agent.last_prediction['extent'].mean()*100):.1f}")

    with tab_limitations:
        st.header("Model Limitations")
        st.markdown("""
        Per SOURCES.md (Rothermel, FlamMap, Sullivan):
        
        - **Surface fire only** (Rothermel 1972): No crown fire or spotting (firebrands).
        - **Quasi-steady state**: Assumes relatively constant conditions over short horizons.
        - **Relative interpretation** (FlamMap): Outputs best for landscape-level comparison, not absolute forecasts.
        - **Educational use** (Sullivan 2009): For research and training; not operational emergency decisions.
        - **Synthetic training**: Initial training on simulated data; real-world validation ongoing.
        
        **Limitations summary:** Approximate model for educational purposes. Validate against real data before deployment.
        """)


if __name__ == "__main__":
    # Check if running with --cli flag
    if "--cli" in sys.argv:
        # Remove --cli from argv so argparse doesn't complain
        sys.argv.remove("--cli")
        run_cli_agent()
    elif HAS_STREAMLIT:
        run_streamlit_app()
    else:
        print("Streamlit not installed. Install with: pip install streamlit")
        print("Or run in CLI mode: python scripts/agent_app.py --cli --checkpoint ...")
