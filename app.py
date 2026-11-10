"""RetinaScope: Diabetic Retinopathy Analysis and Visualization Web Application.

A modern, professional medical imaging application built with Streamlit and PyTorch.
Integrates directly with the trained ResNet-50 research model and Grad-CAM explainability pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

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
    render_explanation,
    render_header,
    render_prediction_card,
    render_probability_bars,
    render_visualizations,
)

# Application paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "best_model_qwk.pth"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "idrid" / "images"

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
def get_cached_model(checkpoint_path_str: str) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], Optional[str]]:
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


def list_sample_images() -> Dict[str, Path]:
    """Discover available external test images in the repository for quick demonstration."""
    sample_dict = {}
    if SAMPLE_IMAGES_DIR.exists():
        for p in sorted(SAMPLE_IMAGES_DIR.glob("external_test_*.jpg"))[:10]:
            name = p.stem.replace("external_test_", "External Benchmark: ")
            sample_dict[name] = p
    return sample_dict


def main() -> None:
    # 1. Model Loading
    model, model_meta, load_error = get_cached_model(str(DEFAULT_CHECKPOINT_PATH))

    # 2. Sidebar Configuration & Status
    with st.sidebar:
        st.markdown("### System & Model Status")
        if model is not None and model_meta is not None:
            st.success("Model Checkpoint Loaded")
            device_name = model_meta.get("device", "cpu").upper()
            st.markdown(f"- **Backbone:** ResNet-50")
            st.markdown(f"- **Device:** `{device_name}`")
            val_qwk = model_meta.get("best_val_qwk")
            if val_qwk is not None:
                st.markdown(f"- **Validation QWK:** `{val_qwk:.4f}`")
            epoch = model_meta.get("epoch")
            if epoch is not None:
                st.markdown(f"- **Best Epoch:** `{epoch}`")
            st.markdown(f"- **Resolution:** `512 x 512`")
        else:
            st.error("Model Not Loaded")
            if load_error:
                st.caption(load_error)

        st.divider()

        st.markdown("### Sample Test Images")
        st.caption("Select a sample fundus image from the external test set or upload your own.")
        sample_options = list_sample_images()
        sample_choice = st.selectbox(
            "Quick Select Benchmark Image",
            options=["-- None (Upload Custom Image) --"] + list(sample_options.keys()),
            index=0,
        )

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
            "Please ensure that the trained PyTorch checkpoint `best_model_qwk.pth` is placed in "
            "the `experiments/checkpoints/` directory before proceeding."
        )
        render_about_section(None)
        return

    # 5. Image Input Section (Upload or Sample selection)
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Retinal Fundus Photograph Input</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([1.6, 1.0])

    selected_image: Optional[Image.Image] = None
    image_source_label = ""

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload Retinal Fundus Photograph",
            type=["jpg", "jpeg", "png"],
            help="Select a color retinal fundus photograph (macula or optic disc centered). Supported formats: JPG, JPEG, PNG.",
        )

    # Resolve image source
    if uploaded_file is not None:
        try:
            selected_image = validate_image(uploaded_file.getvalue())
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Image validation error: {str(e)}")
            selected_image = None
    elif sample_choice != "-- None (Upload Custom Image) --" and sample_choice in sample_options:
        sample_path = sample_options[sample_choice]
        try:
            with Image.open(sample_path) as s_img:
                selected_image = validate_image(s_img)
            image_source_label = f"Benchmark Sample: {sample_path.name}"
        except Exception as e:
            st.error(f"Failed to load sample image: {str(e)}")
            selected_image = None

    with input_col2:
        if selected_image is not None:
            w, h = selected_image.size
            st.image(selected_image, caption=f"{image_source_label} ({w}×{h} px)", use_container_width=True)
        else:
            st.info("Please upload a fundus photograph or choose a sample image from the sidebar.")

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
                # 1. Exact training-time preprocessing
                preprocessed_rgb, tensor_img, meta = preprocess_retinal_image(
                    image=selected_image,
                    target_size=(512, 512),
                    crop_fov=True,
                    device=device,
                )

                # 2. Forward inference
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
                cam_stats = {"coverage_baseline_10pct": 0.0, "peak_coordinate": (0, 0)}

        # Store in session state for persistent display across interactions
        st.session_state["analysis_results"] = {
            "inference": inference_result,
            "preprocessed_rgb": preprocessed_rgb,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "cam_stats": cam_stats,
            "image_source": image_source_label,
        }

    # 8. Render Results
    if "analysis_results" in st.session_state:
        res = st.session_state["analysis_results"]
        inf = res["inference"]
        cam_stats = res["cam_stats"]

        st.divider()

        # Prediction summary and five-class horizontal probability distribution
        col_res_left, col_res_right = st.columns([1.1, 1.0])

        with col_res_left:
            render_prediction_card(inf)

        with col_res_right:
            render_probability_bars(inf["probabilities"], inf["predicted_grade"])

        # Grad-CAM Visualizations
        if res["heatmap_rgb"] is not None and res["overlay_rgb"] is not None:
            render_visualizations(
                preprocessed_img=res["preprocessed_rgb"],
                heatmap_img=res["heatmap_rgb"],
                overlay_img=res["overlay_rgb"],
                stats=cam_stats,
            )

        # Context and Clinical Interpretation
        render_explanation(inf, cam_stats)

    # 9. About Section
    render_about_section(model_meta)


if __name__ == "__main__":
    main()

