"""
Conversational AI agent for wildfire prediction.

Read-only interpretive interface: users ask questions about fire spread predictions,
model confidence, environmental factors, etc. The agent explains model outputs and
limitations based on the trained U-Net and environmental data.

Per PROJECT_SPECIFICATION: "Conversational Agent: Educational, read-only interpretive interface."
Per SOURCES.md: Agent provides context and transparency aligned with scientific sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch


class FirePredictionAgent:
    """
    Interpretive agent for fire spread prediction explanations.

    Provides:
    - Prediction summaries and confidence
    - Environmental factor analysis
    - Comparison to baseline (Rothermel)
    - Model limitations and caveats
    """

    def __init__(
        self,
        model_ckpt: Path,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize agent with trained model.

        Args:
            model_ckpt: Path to trained U-Net checkpoint.
            device: Torch device.
        """
        from src.model import build_unet

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ckpt = model_ckpt

        # Load model
        ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        model_cfg = config.get("model", {})
        in_ch = model_cfg.get("input_channels", 8)
        out_ch = model_cfg.get("output_channels", 1)

        self.model = build_unet(in_channels=in_ch, out_channels=out_ch)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = config
        self.last_prediction: Optional[Dict] = None

    def predict_fire_spread(
        self,
        input_stack: np.ndarray,
    ) -> Dict:
        """
        Run model inference on 8-channel input.

        Args:
            input_stack: (8, H, W) array with channels:
              [slope, aspect_sin, aspect_cos, fuel, wind_speed, wind_sin, wind_cos, initial_fire]

        Returns:
            Dict with 'probability_map', 'extent', 'confidence', 'location_hotspots'.
        """
        x = torch.from_numpy(input_stack).float().unsqueeze(0).to(self.device)  # (1, 8, H, W)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).cpu().numpy().squeeze()  # (H, W)

        extent = (prob > 0.5).astype(np.float32)
        confidence = prob.mean() if prob.size > 0 else 0.0

        # Find high-confidence regions
        hotspots = np.where(prob > 0.7)
        n_hotspots = len(hotspots[0]) if len(hotspots[0]) > 0 else 0

        self.last_prediction = {
            "probability_map": prob,
            "extent": extent,
            "confidence": float(confidence),
            "high_confidence_pixels": n_hotspots,
            "input_stack": input_stack,
        }
        return self.last_prediction

    def analyze_environmental_factors(self) -> str:
        """
        Explain how environmental factors influenced the prediction.

        Returns:
            Narrative explanation of key drivers.
        """
        if self.last_prediction is None:
            return "No prediction yet. Call predict_fire_spread() first."

        stack = self.last_prediction["input_stack"]
        prob = self.last_prediction["probability_map"]

        # Extract normalized factors
        slope = stack[0]
        fuel = stack[3]
        wind_speed = stack[4, 0, 0] * 20.0  # Denormalize

        # High-probability regions
        high_prob_mask = prob > 0.7
        if high_prob_mask.sum() == 0:
            return "Prediction shows low fire probability everywhere."

        avg_slope_high = slope[high_prob_mask].mean()
        avg_fuel_high = fuel[high_prob_mask].mean()

        narrative = (
            f"**Environmental Analysis:**\n"
            f"- Wind speed: {wind_speed:.1f} m/s (estimated from input)\n"
            f"- Average slope in high-probability zones: {avg_slope_high*45:.1f}° "
            f"(normalized: {avg_slope_high:.3f})\n"
            f"- Average fuel density in high-probability zones: {avg_fuel_high:.3f}\n"
            f"- High-probability pixels: {high_prob_mask.sum()}/{prob.size}\n\n"
            f"The model predicts greatest fire extent in regions with higher wind impact, "
            f"moderate to steep slope, and moderate fuel density."
        )
        return narrative

    def compare_to_baseline(self, baseline_extent: np.ndarray) -> str:
        """
        Compare model prediction to Rothermel baseline.

        Args:
            baseline_extent: (H, W) binary array from Rothermel simulator.

        Returns:
            Comparison narrative.
        """
        if self.last_prediction is None:
            return "No prediction yet."

        pred = self.last_prediction["extent"]

        if baseline_extent.shape != pred.shape:
            return "Baseline and prediction have different shapes; cannot compare."

        from src.evaluation import iou_binary, dice_score

        iou = iou_binary(pred, baseline_extent)
        dice = dice_score(pred, baseline_extent)

        agreement = (pred == baseline_extent).sum() / pred.size

        narrative = (
            f"**Comparison to Rothermel Baseline:**\n"
            f"- IoU (Intersection over Union): {iou:.4f}\n"
            f"- Dice coefficient: {dice:.4f}\n"
            f"- Pixel agreement: {agreement*100:.1f}%\n"
            f"\nThe model prediction aligns with the Rothermel baseline in "
            f"{agreement*100:.0f}% of pixels. Differences may reflect the model's "
            f"ability to capture non-linear fire behavior beyond the deterministic baseline."
        )
        return narrative

    def explain_limitations(self) -> str:
        """Return structured explanation of model limitations."""
        return (
            "**Model Limitations (Per SOURCES.md and FlamMap):**\n"
            "1. **Surface fire only**: Predictions apply to surface fire spread, not crown fire or spotting.\n"
            "2. **Quasi-steady state**: Assumes relatively constant environmental conditions over short horizons.\n"
            "3. **Synthetic training data**: Initial training uses simulated data; real-world accuracy TBD.\n"
            "4. **Relative sense interpretation**: Outputs are best used for landscape-level comparison "
            "(e.g., treatment effectiveness), not as absolute forecasts.\n"
            "5. **Local scope**: Model trained on ~60–90 m grid; predictions designed for regional (~10 km²) analysis.\n"
            "6. **Educational use only**: Not for emergency decision-making without expert validation."
        )

    def answer_question(self, question: str) -> str:
        """
        Answer a user question about the prediction.

        Supported question types:
        - "What is the predicted fire extent?"
        - "Which areas are most at risk?"
        - "How confident is the model?"
        - "How does this compare to the baseline?"
        - "What are the limitations?"
        - "What environmental factors drive this?"

        Args:
            question: User's question.

        Returns:
            Answer string.
        """
        q_lower = question.lower()

        if not self.last_prediction:
            return "No prediction loaded. Please provide input data and call predict_fire_spread() first."

        # Dispatch
        if any(x in q_lower for x in ["extent", "spread", "burn area", "affected area"]):
            prob = self.last_prediction["probability_map"]
            extent = self.last_prediction["extent"]
            affected_pct = extent.mean() * 100
            return (
                f"**Predicted Fire Extent:**\n"
                f"Approximately {affected_pct:.1f}% of the grid may be affected by fire.\n"
                f"High-confidence (>70%) fire: {(prob > 0.7).sum()} pixels.\n"
                f"Moderate-confidence (50-70%) fire: {((prob > 0.5) & (prob <= 0.7)).sum()} pixels."
            )

        elif any(x in q_lower for x in ["risk", "hotspot", "dangerous", "worst"]):
            prob = self.last_prediction["probability_map"]
            hotspot_mask = prob > 0.7
            if hotspot_mask.sum() == 0:
                return "No high-risk zones identified (probability <70% everywhere)."
            return (
                f"**High-Risk Zones:**\n"
                f"{hotspot_mask.sum()} grid cells show >70% fire probability.\n"
                f"Mean probability in hotspots: {prob[hotspot_mask].mean():.3f}.\n"
                f"These regions warrant closer attention in fire management planning."
            )

        elif any(x in q_lower for x in ["confident", "confidence", "certain", "sure"]):
            conf = self.last_prediction["confidence"]
            return (
                f"**Model Confidence:**\n"
                f"Mean fire probability: {conf:.3f} (scale 0–1).\n"
                f"{'High confidence.' if conf > 0.6 else 'Moderate confidence.' if conf > 0.3 else 'Low confidence.'}\n"
                f"This reflects the average probability across the predicted area."
            )

        elif any(x in q_lower for x in ["limitation", "caveat", "assumption", "constraint"]):
            return self.explain_limitations()

        elif any(x in q_lower for x in ["baseline", "compare", "rothermel"]):
            return "Please provide the baseline extent to compare."

        elif any(x in q_lower for x in ["environment", "factor", "wind", "slope", "fuel", "driver"]):
            return self.analyze_environmental_factors()

        else:
            return (
                "I can help explain:\n"
                "- Predicted fire extent and spread\n"
                "- High-risk zones\n"
                "- Model confidence\n"
                "- Environmental factors\n"
                "- Comparison to baseline\n"
                "- Model limitations\n\n"
                "Ask a question about these topics, or provide more details."
            )


if __name__ == "__main__":
    print("FirePredictionAgent: Educational fire spread prediction interface.")
    print("Usage:")
    print("  agent = FirePredictionAgent('path/to/checkpoint.pt')")
    print("  results = agent.predict_fire_spread(input_stack)")
    print("  agent.answer_question('What is the predicted fire extent?')")
