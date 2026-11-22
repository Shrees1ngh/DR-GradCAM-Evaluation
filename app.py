"""RetinaScope: Clean & Intuitive Diabetic Retinopathy Diagnostic Application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app.benchmark import (
    evaluate_live_localization,
    load_benchmark_catalogs,
    load_expert_lesion_data,
    resolve_image_path,
)
from app.gradcam import compute_gradcam
from app.inference import (
    get_device,
    load_trained_model,
    preprocess_retinal_image,
    run_inference,
    validate_image,
)
import importlib
import app.ui as ui

# Hot-reload guard: force-refresh ui module from disk on every rerun
try:
    importlib.reload(ui)
except Exception:
    pass

get_custom_css = ui.get_custom_css
render_clinical_recommendation = ui.render_clinical_recommendation
render_header = ui.render_header
render_prediction_card = ui.render_prediction_card
render_probability_bars = ui.render_probability_bars
render_visualizations = ui.render_visualizations
render_xai_reasoning = ui.render_xai_reasoning
render_localization_benchmark = getattr(ui, "render_localization_benchmark", None)

# Application paths
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "model_b_best.pth"

# Page configuration
st.set_page_config(
    page_title="RetinaScope | DR Diagnostic Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path_str: str) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], Optional[str]]:
    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.exists():
        return None, None, f"Checkpoint not found at: {ckpt_path}"
    try:
        device = get_device()
        model, meta = load_trained_model(ckpt_path, device=device)
        return model, meta, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False)
def get_cached_benchmarks() -> Dict[str, pd.DataFrame]:
    return load_benchmark_catalogs(PROJECT_ROOT)


def discover_checkpoints() -> Dict[str, Path]:
    available: Dict[str, Path] = {}
    model_b = CHECKPOINTS_DIR / "model_b_best.pth"
    model_a = CHECKPOINTS_DIR / "best_model_qwk.pth"
    if model_b.exists():
        available["Model B (Fine-Tuned — Recommended)"] = model_b
    if model_a.exists():
        available["Model A (APTOS Baseline)"] = model_a
    return available


def get_sample_cases(benchmarks: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    cases = []
    if "IDRiD External Test (103 images)" in benchmarks:
        df = benchmarks["IDRiD External Test (103 images)"]
        samples = [
            (0, "Normal Retina (Grade 0)"),
            (1, "Mild NPDR (Grade 1 — Microaneurysms)"),
            (2, "Moderate NPDR (Grade 2 — Hemorrhages/Exudates)"),
            (3, "Severe NPDR (Grade 3 — Extensive Hemorrhages)"),
            (4, "Proliferative DR (Grade 4 — Neovascularization)"),
        ]
        for grade, name in samples:
            sub = df[df["true_grade"] == grade]
            if len(sub) > 0:
                row = sub.iloc[0]
                cases.append({
                    "name": name,
                    "dataset": "IDRiD External Test (103 images)",
                    "row": row,
                })

    if "IDRiD Segmentation (81 lesion-masked images)" in benchmarks:
        df_seg = benchmarks["IDRiD Segmentation (81 lesion-masked images)"]
        pos = df_seg[df_seg["composite_pixels"] > 0]
        if len(pos) > 0:
            cases.append({
                "name": "IDRiD Lesion-Masked Case (Grade 2 with Segmentation)",
                "dataset": "IDRiD Segmentation (81 lesion-masked images)",
                "row": pos.iloc[0],
            })
    return cases


def main() -> None:
    # 1. Discover Checkpoints & Model
    available_ckpts = discover_checkpoints()
    if available_ckpts:
        selected_label = st.sidebar.selectbox("Model Checkpoint", list(available_ckpts.keys()), index=0)
        ckpt_path = available_ckpts[selected_label]
    else:
        ckpt_path = DEFAULT_CHECKPOINT_PATH

    model, model_meta, load_error = get_cached_model(str(ckpt_path))
    benchmarks = get_cached_benchmarks()

    # 2. Clean Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        if model is not None:
            st.success("● Model Loaded & Ready")
            st.caption(f"**Backbone:** ResNet-50  \n**Validation QWK:** 0.8915  \n**Resolution:** 512×512")
        else:
            st.error("Model Offline")
            if load_error:
                st.caption(load_error)

        st.divider()

        if st.button("🔄 Reset Analysis", use_container_width=True):
            if "analysis_results" in st.session_state:
                del st.session_state["analysis_results"]
            st.rerun()

        st.caption(
            "---  \n"
            "**Academic Research Project**  \n"
            "*Research prototype; not approved for clinical medical use.*"
        )

    # 3. Main Header
    render_header()

    if model is None:
        st.error(f"Checkpoint `model_b_best.pth` not found at `{DEFAULT_CHECKPOINT_PATH}`.")
        return

    # 4. Clean Input Section
    with st.container(border=True):
        st.markdown("#### Select or Upload a Retinal Fundus Photograph")
        
        input_choice = st.radio(
            "Input Mode",
            ["Choose a Sample Retinal Image", "Upload Custom Image"],
            horizontal=True,
            label_visibility="collapsed",
        )

        selected_image: Optional[Image.Image] = None
        image_label = ""
        active_row: Optional[pd.Series] = None
        active_dataset = ""

        if input_choice == "Choose a Sample Retinal Image":
            samples = get_sample_cases(benchmarks)
            if samples:
                sample_names = [s["name"] for s in samples]
                chosen_sample = st.selectbox("Select Sample Image:", sample_names, index=2)
                matched = next(s for s in samples if s["name"] == chosen_sample)
                active_row = matched["row"]
                active_dataset = matched["dataset"]
                p = resolve_image_path(active_row, PROJECT_ROOT)
                try:
                    with Image.open(p) as img:
                        selected_image = validate_image(img)
                    image_label = f"{matched['name']} ({p.name})"
                except Exception as e:
                    st.error(f"Failed to load sample: {e}")
            else:
                st.info("No local benchmark datasets found. Please upload an image.")
        else:
            uploaded = st.file_uploader(
                "Upload Fundus Photograph",
                type=["jpg", "jpeg", "png"],
                help="Accepts high-resolution fundus images. Auto-scales to 512×512.",
            )
            if uploaded is not None:
                try:
                    selected_image = validate_image(uploaded.getvalue())
                    image_label = f"Uploaded: {uploaded.name}"
                except Exception as e:
                    st.error(f"Image error: {e}")

        if selected_image is not None:
            col_preview, col_action = st.columns([1, 1], gap="large", vertical_alignment="center")
            with col_preview:
                st.image(selected_image, caption=image_label, use_container_width=True)
            with col_action:
                w, h = selected_image.size
                st.markdown("##### Retinal Scan Ready")
                st.caption(f"• **Resolution:** {w} × {h} px  \n• **Input Pipeline:** 512 × 512 FOV  \n• **Color Channels:** 3-Channel RGB")
                run_btn = st.button("🔬 Analyze Retinal Image", type="primary", use_container_width=True)
        else:
            run_btn = False

    # 5. Process Inference
    if run_btn and selected_image is not None:
        device = get_device()
        with st.spinner("Processing fundus image through ResNet-50 & generating Grad-CAM..."):
            try:
                preprocessed_rgb, tensor_img, _ = preprocess_retinal_image(
                    image=selected_image,
                    target_size=(512, 512),
                    crop_fov=True,
                    device=device,
                )
                inf_result = run_inference(model=model, tensor_img=tensor_img)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

            try:
                heatmap_rgb, overlay_rgb, grayscale_cam, cam_stats = compute_gradcam(
                    model=model,
                    tensor_img=tensor_img,
                    preprocessed_rgb=preprocessed_rgb,
                    target_grade=inf_result["predicted_grade"],
                )
            except Exception as e:
                st.warning(f"Grad-CAM error: {e}")
                heatmap_rgb, overlay_rgb, grayscale_cam = None, None, None
                cam_stats = {"coverage_baseline_10pct": 0.0, "peak_coordinate": (0, 0)}

            # Optional benchmark metrics
            true_grade = None
            loc_metrics = None
            expert_data = None
            if active_row is not None:
                tg = active_row.get("true_grade")
                if pd.notna(tg):
                    true_grade = int(tg)
                if "Segmentation" in active_dataset and grayscale_cam is not None:
                    expert_data = load_expert_lesion_data(active_row, preprocessed_rgb, PROJECT_ROOT)
                    if expert_data is not None:
                        loc_metrics = evaluate_live_localization(grayscale_cam, expert_data)

            st.session_state["analysis_results"] = {
                "inference": inf_result,
                "preprocessed_rgb": preprocessed_rgb,
                "heatmap_rgb": heatmap_rgb,
                "overlay_rgb": overlay_rgb,
                "cam_stats": cam_stats,
                "true_grade": true_grade,
                "loc_metrics": loc_metrics,
                "expert_data": expert_data,
            }

    # 6. Clean Results Section
    if "analysis_results" in st.session_state:
        res = st.session_state["analysis_results"]
        inf = res["inference"]
        cam_stats = res["cam_stats"]

        st.subheader("Diagnostic Results")

        # Diagnosis Banner
        render_prediction_card(inf)

        # Ground truth match if available
        if res.get("true_grade") is not None:
            tg = res["true_grade"]
            pred = inf["predicted_grade"]
            is_match = tg == pred
            match_color = "green" if is_match else ("orange" if abs(tg - pred) <= 1 else "red")
            match_text = "Exact Match" if is_match else ("Within 1 Grade" if abs(tg - pred) <= 1 else "Discrepancy")
            st.info(
                f"**Benchmark Reference:** Ground-Truth Grade is **Grade {tg}** | "
                f"Prediction Alignment: **:{match_color}[{match_text}]**"
            )

        # Two-column balanced results
        col_rec, col_probs = st.columns([1, 1], gap="medium")
        with col_rec:
            render_clinical_recommendation(inf)
        with col_probs:
            render_probability_bars(inf["probabilities"], inf["predicted_grade"])

        # Saliency Visualizations
        if res["heatmap_rgb"] is not None and res["overlay_rgb"] is not None:
            render_visualizations(
                preprocessed_img=res["preprocessed_rgb"],
                heatmap_img=res["heatmap_rgb"],
                overlay_img=res["overlay_rgb"],
                stats=cam_stats,
            )

        # Plain-English Explainable AI (XAI) Reasoning
        render_xai_reasoning(inf, cam_stats)

        # Live Lesion Localization Benchmark (when expert mask is available)
        if (
            render_localization_benchmark is not None
            and res.get("loc_metrics") is not None
            and res.get("expert_data") is not None
        ):
            render_localization_benchmark(
                loc_metrics=res["loc_metrics"],
                expert_data=res["expert_data"],
                preprocessed_rgb=res["preprocessed_rgb"],
            )


if __name__ == "__main__":
    main()
