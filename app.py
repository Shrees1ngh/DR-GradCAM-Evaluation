"""RetinaScope: Diabetic Retinopathy Analysis and Visualization Web Application.

A modern, professional medical imaging application built with Streamlit and PyTorch.
Integrates directly with the trained ResNet-50 research model (Model B: model_b_best.pth)
and Grad-CAM explainability pipeline using genuine data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
from app.ui import (
    get_custom_css,
    render_about_section,
    render_benchmark_reference,
    render_explanation,
    render_header,
    render_prediction_card,
    render_probability_bars,
    render_visualizations,
)

# Application paths
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "model_b_best.pth"

# Configure Streamlit page
st.set_page_config(
    page_title="RetinaScope | Diabetic Retinopathy Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply medical research CSS design
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model(
    checkpoint_path_str: str,
) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], Optional[str]]:
    """Load and cache the trained model to avoid reloading on every interaction."""
    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.exists():
        return None, None, f"Checkpoint file not found at: {ckpt_path}"

    try:
        device = get_device()
        model, meta = load_trained_model(ckpt_path, device=device)
        return model, meta, None
    except Exception as e:
        return None, None, f"Failed to initialize model from checkpoint: {str(e)}"


@st.cache_data(show_spinner=False)
def get_cached_benchmarks() -> Dict[str, pd.DataFrame]:
    """Load real dataset manifests if available locally."""
    return load_benchmark_catalogs(PROJECT_ROOT)


def discover_checkpoints() -> Dict[str, Path]:
    """Discover available model checkpoints, prioritizing model_b_best.pth."""
    available: Dict[str, Path] = {}
    model_b = CHECKPOINTS_DIR / "model_b_best.pth"
    model_a = CHECKPOINTS_DIR / "best_model_qwk.pth"

    if model_b.exists():
        available["Model B (model_b_best.pth — Latest IDRiD Fine-Tuned)"] = model_b
    if model_a.exists():
        available["Model A (best_model_qwk.pth — APTOS Baseline)"] = model_a
    return available


def main() -> None:
    # 1. Discover Checkpoints & Load Active Model
    available_ckpts = discover_checkpoints()
    if available_ckpts:
        selected_ckpt_label = st.sidebar.selectbox(
            "Active Model Checkpoint",
            options=list(available_ckpts.keys()),
            index=0,
        )
        active_ckpt_path = available_ckpts[selected_ckpt_label]
    else:
        active_ckpt_path = DEFAULT_CHECKPOINT_PATH

    model, model_meta, load_error = get_cached_model(str(active_ckpt_path))
    benchmarks = get_cached_benchmarks()

    # 2. Sidebar Configuration & Status
    with st.sidebar:
        st.markdown("### System & Model Status")
        if model is not None and model_meta is not None:
            st.success("Model Checkpoint Loaded")
            device_name = model_meta.get("device", "cpu").upper()
            ckpt_name = model_meta.get("checkpoint_name", active_ckpt_path.name)
            st.markdown(f"- **Checkpoint:** `{ckpt_name}`")
            st.markdown(f"- **Backbone:** ResNet-50")
            st.markdown(f"- **Device:** `{device_name}`")
            val_qwk = model_meta.get("best_val_qwk")
            if val_qwk is not None:
                st.markdown(f"- **Validation QWK:** `{val_qwk:.4f}`")
            epoch = model_meta.get("epoch")
            if epoch is not None:
                st.markdown(f"- **Best Epoch:** `{epoch}`")
            st.markdown("- **Resolution:** `512 x 512`")
        else:
            st.error("Model Not Loaded")
            if load_error:
                st.caption(load_error)

        st.divider()

        st.markdown("### Real Benchmark Dataset")
        selected_row: Optional[pd.Series] = None
        selected_dataset_name = "-- None (Upload Custom Image) --"

        if benchmarks:
            selected_dataset_name = st.selectbox(
                "Dataset Cohort",
                options=["-- None (Upload Custom Image) --"] + list(benchmarks.keys()),
                index=0,
            )
            if selected_dataset_name != "-- None (Upload Custom Image) --":
                df_bench = benchmarks[selected_dataset_name]
                # Build readable option labels
                option_labels = []
                for idx, r in df_bench.iterrows():
                    oid = str(r.get("original_id", r.get("image_id", idx)))
                    tg = r.get("true_grade")
                    grade_str = f" | Grade {int(tg)}" if pd.notna(tg) else ""
                    option_labels.append(f"{oid}{grade_str}")

                chosen_idx = st.selectbox(
                    "Select Retinal Image",
                    options=list(range(len(df_bench))),
                    format_func=lambda i: option_labels[i],
                )
                selected_row = df_bench.iloc[chosen_idx]
        else:
            st.caption("No local processed datasets detected. Upload a fundus image directly.")

        st.divider()

        if st.button("Reset Analysis", use_container_width=True):
            if "analysis_results" in st.session_state:
                del st.session_state["analysis_results"]
            st.rerun()

    # 3. Main Header
    render_header(model_meta)

    # 4. Checkpoint Error Gate
    if model is None:
        st.error(
            "⚠️ **Required Model Checkpoint Missing or Inaccessible**\n\n"
            f"Expected location: `{DEFAULT_CHECKPOINT_PATH}`\n\n"
            "RetinaScope strictly prohibits mock or fabricated predictions. "
            "Please ensure that the trained PyTorch checkpoint `model_b_best.pth` is placed in "
            "the `experiments/checkpoints/` directory before proceeding."
        )
        render_about_section(None)
        return

    # 5. Image Input Section (Upload or Real Benchmark Selection)
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Retinal Fundus Photograph Input</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([1.6, 1.0])

    selected_image: Optional[Image.Image] = None
    image_source_label = ""
    active_benchmark_row: Optional[pd.Series] = None
    active_benchmark_dataset = ""

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload Retinal Fundus Photograph",
            type=["jpg", "jpeg", "png"],
            help="Select a color retinal fundus photograph (macula or optic disc centered). Supported formats: JPG, JPEG, PNG.",
        )

    # Uploaded file takes precedence over sidebar benchmark selection
    if uploaded_file is not None:
        try:
            selected_image = validate_image(uploaded_file.getvalue())
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Image validation error: {str(e)}")
            selected_image = None
    elif selected_row is not None:
        img_path = resolve_image_path(selected_row, PROJECT_ROOT)
        try:
            with Image.open(img_path) as s_img:
                selected_image = validate_image(s_img)
            image_source_label = f"{selected_dataset_name}: {img_path.name}"
            active_benchmark_row = selected_row
            active_benchmark_dataset = selected_dataset_name
        except Exception as e:
            st.error(f"Failed to load benchmark image ({img_path}): {str(e)}")
            selected_image = None

    with input_col2:
        if selected_image is not None:
            w, h = selected_image.size
            st.image(selected_image, caption=f"{image_source_label} ({w}×{h} px)", use_container_width=True)
        else:
            st.info("Please upload a fundus photograph or choose a real image from the sidebar dataset browser.")

    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Action Button
    analyze_clicked = False
    if selected_image is not None:
        analyze_clicked = st.button("🔬 Analyze Retinal Image", use_container_width=False)

    # 7. Processing Pipeline
    if analyze_clicked and selected_image is not None:
        device = get_device()

        with st.spinner("Executing isotropic FOV preprocessing and neural network inference..."):
            try:
                preprocessed_rgb, tensor_img, _ = preprocess_retinal_image(
                    image=selected_image,
                    target_size=(512, 512),
                    crop_fov=True,
                    device=device,
                )
                inference_result = run_inference(model=model, tensor_img=tensor_img)
            except Exception as e:
                st.error(f"Inference failed during computation: {str(e)}")
                return

        with st.spinner("Generating Grad-CAM salience visualization from ResNet-50 layer4..."):
            try:
                heatmap_rgb, overlay_rgb, grayscale_cam, cam_stats = compute_gradcam(
                    model=model,
                    tensor_img=tensor_img,
                    preprocessed_rgb=preprocessed_rgb,
                    target_grade=inference_result["predicted_grade"],
                )
            except Exception as e:
                st.warning(f"Grad-CAM generation encountered an issue: {str(e)}")
                heatmap_rgb = None
                overlay_rgb = None
                grayscale_cam = None
                cam_stats = {"coverage_baseline_10pct": 0.0, "peak_coordinate": (0, 0)}

        # Post-inference reference evaluation (only when a real benchmark image is selected)
        true_grade = None
        expert_data = None
        loc_metrics = None
        if active_benchmark_row is not None:
            tg = active_benchmark_row.get("true_grade")
            if pd.notna(tg):
                true_grade = int(tg)
            if "Segmentation" in active_benchmark_dataset and grayscale_cam is not None:
                expert_data = load_expert_lesion_data(active_benchmark_row, preprocessed_rgb, PROJECT_ROOT)
                if expert_data is not None:
                    loc_metrics = evaluate_live_localization(grayscale_cam, expert_data)

        st.session_state["analysis_results"] = {
            "inference": inference_result,
            "preprocessed_rgb": preprocessed_rgb,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "cam_stats": cam_stats,
            "image_source": image_source_label,
            "true_grade": true_grade,
            "expert_data": expert_data,
            "loc_metrics": loc_metrics,
        }

    # 8. Render Results
    if "analysis_results" in st.session_state:
        res = st.session_state["analysis_results"]
        inf = res["inference"]
        cam_stats = res["cam_stats"]

        st.divider()

        col_res_left, col_res_right = st.columns([1.1, 1.0])
        with col_res_left:
            render_prediction_card(inf)
        with col_res_right:
            render_probability_bars(inf["probabilities"], inf["predicted_grade"])

        if res["heatmap_rgb"] is not None and res["overlay_rgb"] is not None:
            render_visualizations(
                preprocessed_img=res["preprocessed_rgb"],
                heatmap_img=res["heatmap_rgb"],
                overlay_img=res["overlay_rgb"],
                stats=cam_stats,
            )

        render_benchmark_reference(
            true_grade=res.get("true_grade"),
            predicted_grade=inf["predicted_grade"],
            expert_data=res.get("expert_data"),
            loc_metrics=res.get("loc_metrics"),
            preprocessed_rgb=res.get("preprocessed_rgb"),
        )

        render_explanation(inf, cam_stats)

    # 9. About Section
    render_about_section(model_meta)


if __name__ == "__main__":
    main()
