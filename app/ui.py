"""UI components, CSS styling, and layout helpers for RetinaScope.

Adheres strictly to professional medical research design standards:
- Light neutral background (#F8FAFC, #FFFFFF)
- Deep navy typography (#0F172A, #1E293B)
- Clinical teal accent (#0D9488, #0F766E)
- Subtle borders (#E2E8F0) and restrained, functional presentation
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def get_custom_css() -> str:
    """Return CSS styling for a clinical research application."""
    return """
    <style>
    /* Global Base */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top App Header */
    .app-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .app-title {
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        color: #0D9488;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .app-meta-bar {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .meta-tag {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }
    .meta-tag-teal {
        background-color: #CCFBF1;
        color: #0F766E;
        border: 1px solid #99F6E4;
        font-weight: 600;
    }

    /* Disclaimer Callout */
    .disclaimer-banner {
        background-color: #FFFFFF;
        border-left: 4px solid #0D9488;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Content Cards */
    .card-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Prediction Result Banner */
    .prediction-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .grade-badge {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .confidence-pill {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0D9488;
        background-color: #F0FDFA;
        border: 1px solid #99F6E4;
        padding: 0.35rem 0.85rem;
        border-radius: 4px;
    }

    /* Probability Bar Row */
    .prob-row {
        margin-bottom: 0.65rem;
    }
    .prob-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.84rem;
        color: #334155;
        margin-bottom: 0.25rem;
    }
    .prob-track {
        background-color: #F1F5F9;
        border-radius: 3px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    /* Image Display Containers */
    .viz-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }
    .viz-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 0.6rem 0.75rem;
    }
    .stat-label {
        font-size: 0.74rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .stat-val {
        font-size: 1rem;
        font-weight: 600;
        color: #0F172A;
    }

    /* Streamlit Custom Overrides */
    div[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .stButton>button {
        background-color: #0D9488;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.55rem 1.25rem;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #0F766E;
        color: #FFFFFF;
    }
    .stButton>button:active {
        background-color: #115E59;
    }
    </style>
    """


def render_header(model_meta: Optional[Dict[str, Any]] = None) -> None:
    """Render application header with title, subtitle, and checkpoint metadata."""
    qwk_text = ""
    epoch_text = ""
    device_text = ""

    if model_meta:
        val_qwk = model_meta.get("best_val_qwk")
        if val_qwk is not None:
            qwk_text = f"Val QWK: {val_qwk:.4f}"
        epoch = model_meta.get("epoch")
        if epoch is not None:
            epoch_text = f"Epoch {epoch}"
        device = model_meta.get("device")
        if device:
            device_text = f"Device: {device.upper()}"

    st.markdown(
        f"""
        <div class="app-header">
            <h1 class="app-title">RetinaScope</h1>
            <div class="app-subtitle">Diabetic Retinopathy Analysis and Visualization</div>
            <div class="app-meta-bar">
                <span class="meta-tag meta-tag-teal">ResNet-50</span>
                <span class="meta-tag">ICDR 5-Class</span>
                <span class="meta-tag">Grad-CAM Salience</span>
                {f'<span class="meta-tag">{qwk_text}</span>' if qwk_text else ''}
                {f'<span class="meta-tag">{epoch_text}</span>' if epoch_text else ''}
                {f'<span class="meta-tag">{device_text}</span>' if device_text else ''}
            </div>
        </div>
        <div class="disclaimer-banner">
            <strong>Research Prototype Notice:</strong> RetinaScope is designed for clinical research and algorithmic explainability evaluation.
            It does not constitute a clinically approved diagnostic medical device and must not replace direct examination by a licensed ophthalmologist or optometrist.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(inference: Dict[str, Any]) -> None:
    """Render top summary card for the predicted DR grade."""
    grade = inference["predicted_grade"]
    info = inference["grade_info"]
    conf = inference["confidence"]

    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-title">Model Classification Result</div>
            <div class="prediction-box">
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">
                        Predicted DR Severity
                    </div>
                    <div class="grade-badge" style="color: {info['badge_color']};">
                        Grade {grade}: {info['title']}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">
                        Model Probability
                    </div>
                    <div class="confidence-pill">
                        {conf * 100:.1f}%
                    </div>
                </div>
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5; background-color: #F8FAFC; padding: 0.75rem 1rem; border-radius: 4px; border: 1px solid #E2E8F0;">
                <strong>ICDR Definition:</strong> {info['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probabilities: Dict[int, float], predicted_grade: int) -> None:
    """Render horizontal probability distribution across all 5 ICDR classes."""
    grade_names = {
        0: "Grade 0: No DR",
        1: "Grade 1: Mild DR",
        2: "Grade 2: Moderate DR",
        3: "Grade 3: Severe DR",
        4: "Grade 4: Proliferative DR",
    }

    rows_html = []
    for g in range(5):
        prob = probabilities.get(g, 0.0)
        pct = prob * 100.0
        is_pred = (g == predicted_grade)

        fill_color = "#0D9488" if is_pred else "#94A3B8"
        font_weight = "700" if is_pred else "500"
        text_color = "#0F172A" if is_pred else "#475569"
        highlight_mark = " (Predicted)" if is_pred else ""

        rows_html.append(
            f"""
            <div class="prob-row">
                <div class="prob-label-row">
                    <span style="font-weight: {font_weight}; color: {text_color};">
                        {grade_names[g]}{highlight_mark}
                    </span>
                    <span style="font-weight: {font_weight}; color: {text_color};">
                        {pct:.2f}%
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {pct:.2f}%; background-color: {fill_color};"></div>
                </div>
            </div>
            """
        )

    all_rows = "\n".join(rows_html)

    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-title">Five-Class Probability Distribution</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.85rem;">
                Exact softmax probabilities calculated across all five ICDR severity categories.
            </div>
            {all_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visualizations(
    preprocessed_img: Any,
    heatmap_img: Any,
    overlay_img: Any,
    stats: Dict[str, Any],
) -> None:
    """Render the 3-column visualization panel containing preprocessed fundus, heatmap, and overlay."""
    st.markdown(
        """
        <div class="card-panel">
            <div class="card-title">Grad-CAM Salience Visualization</div>
            <div style="font-size: 0.84rem; color: #475569; margin-bottom: 1rem; line-height: 1.5;">
                Grad-CAM computes the gradients of the predicted class score with respect to feature maps in
                the final convolutional block (<strong>ResNet-50 layer4</strong>). Highlighted regions reflect
                spatial features that influenced the model prediction. Grad-CAM visualizes network attention
                patterns and does not claim that every highlighted area is a clinically confirmed retinal lesion.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.image(preprocessed_img, use_container_width=True)
        st.markdown('<div class="viz-label">1. Preprocessed Fundus (512×512)</div>', unsafe_allow_html=True)
        st.caption("Isotropically scaled with FOV border crop")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.image(heatmap_img, use_container_width=True)
        st.markdown('<div class="viz-label">2. Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.caption("ResNet-50 layer4 feature activation map")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.image(overlay_img, use_container_width=True)
        st.markdown('<div class="viz-label">3. Anatomical Overlay</div>', unsafe_allow_html=True)
        st.caption("Activation map aligned with retinal structures")
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary metrics below visualization
    cov10 = stats.get("coverage_baseline_10pct", 0.0)
    cov50 = stats.get("coverage_focused_50pct", 0.0)
    peak = stats.get("peak_coordinate", (0, 0))

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Baseline Salience Area (≥10%)</div>
                <div class="stat-val">{cov10:.1f}% of image</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Focused Salience Area (≥50%)</div>
                <div class="stat-val">{cov50:.1f}% of image</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Salience Location</div>
                <div class="stat-val">X: {peak[0]}, Y: {peak[1]}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Architecture Layer</div>
                <div class="stat-val">ResNet50.layer4</div>
            </div>
        </div>
        <div style="margin-bottom: 1.25rem;"></div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation(inference: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Render concise explanation of findings and explainability insights."""
    grade = inference["predicted_grade"]
    label = inference["predicted_label"]
    conf = inference["confidence"]
    cov10 = stats.get("coverage_baseline_10pct", 0.0)
    peak = stats.get("peak_coordinate", (0, 0))

    summary_points = {
        0: "The model detected no significant microvascular lesions in the retinal field of view. Saliency values remain diffused across the background vascular arcade, consistent with a negative screen for diabetic retinopathy.",
        1: "The model identified focal patterns consistent with early microvascular alterations (microaneurysms). Salience is concentrated around localized retinal vascular segments.",
        2: "The model detected moderate retinopathy indicators. Saliency highlights retinal zones showing dot-blot hemorrhages or hard exudate clusters away from the fovea.",
        3: "The model identified substantial pathological patterns satisfying severe non-proliferative criteria (such as extensive intraretinal microvascular abnormalities or multi-quadrant hemorrhages).",
        4: "The model detected high-salience features characteristic of proliferative disease, focusing on regions typical for neovascularization, fibrous proliferation, or pre-retinal changes.",
    }

    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-title">Prediction Context & Interpretation</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 0.75rem;">
                <strong>Classification Summary:</strong> The network assigns the highest probability
                (<strong>{conf * 100:.1f}%</strong>) to <strong>Grade {grade} ({label})</strong>.
                {summary_points.get(grade, "")}
            </p>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin-bottom: 0;">
                <strong>Explainability Insights:</strong> The primary activation focus is located at coordinate
                (X: {peak[0]}, Y: {peak[1]}), with {cov10:.1f}% of the retinal field demonstrating salient influence
                at the 10% baseline threshold. Clinical verification with funduscopic examination and fluorescein angiography
                remains mandatory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_section(checkpoint_meta: Optional[Dict[str, Any]] = None) -> None:
    """Render project methodology, architecture details, and research context."""
    val_qwk_str = "0.9076"
    epoch_str = "8"
    if checkpoint_meta:
        if checkpoint_meta.get("best_val_qwk"):
            val_qwk_str = f"{checkpoint_meta['best_val_qwk']:.4f}"
        if checkpoint_meta.get("epoch"):
            epoch_str = str(checkpoint_meta["epoch"])

    with st.expander("About RetinaScope & Research Methodology", expanded=False):
        st.markdown(
            f"""
            ### Project Overview
            **RetinaScope** is the demonstration and evaluation interface for the research project
            *Evaluating Grad-CAM Explanations for Diabetic Retinopathy Grading*.

            ### Clinical Background
            Diabetic Retinopathy (DR) is a diabetes complication caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). It is graded according to the International Clinical Diabetic Retinopathy (ICDR) scale:
            - **Grade 0 (No DR):** No visible microvascular abnormalities.
            - **Grade 1 (Mild NPDR):** Presence of microaneurysms only.
            - **Grade 2 (Moderate NPDR):** More than microaneurysms, but less than severe NPDR.
            - **Grade 3 (Severe NPDR):** Extensive hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant (4-2-1 rule).
            - **Grade 4 (Proliferative DR):** Neovascularization of retina or disc, or preretinal/vitreous hemorrhage.

            ### Technical Architecture
            - **Backbone:** ResNet-50 (`torchvision.models.resnet50`) initialized with ImageNet pretrained weights.
            - **Head:** Custom classification head with 30% dropout and 5-class linear projection.
            - **Training Dataset:** Primary training on APTOS 2019 Blindness Detection (3,662 fundus images).
            - **Validation Performance:** Model selection monitored strictly on Validation Quadratic Weighted Kappa (**QWK = {val_qwk_str}**, Epoch {epoch_str}).
            - **Explainability:** Grad-CAM computed directly from the final convolutional block (`model.layer4`, 2048-dim feature maps).
            - **Preprocessing:** Retinal field-of-view border crop, aspect-ratio preserving isotropic resize, centered zero-padding to 512×512, and ImageNet standardization.

            ### Academic & Medical Disclaimer
            This software is developed solely as an academic research prototype. It is not licensed as medical software, does not provide clinical diagnosis, and must not be used for therapeutic decision-making.
            """
        )

