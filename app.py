import streamlit as st
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import io
import torch
import base64

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DetectionModel])

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vehicle Detection Demo",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
MODEL_FILES = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')] if os.path.exists(MODELS_DIR) else []
SAMPLE_IMAGES = [f for f in os.listdir(ASSETS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(ASSETS_DIR) else []
CONFIDENCE_THRESHOLD = 0.55
DISPLAY_CONFIDENCE_THRESHOLD = 0.75
MIN_BOX_AREA_RATIO = 0.015

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
    <style>
        :root {
            --bg: #F6F1E8;
            --card: #FFFDF8;
            --text: #2B2622;
            --text-muted: #6A5E52;
            --border: #DED3C4;
            --accent: #C86B36;
            --accent-hover: #A9552D;
            --success: #667A3E;
            --error: #B14E3A;
        }

        html, body, [class*="st-"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        .stApp { background-color: var(--bg); color: var(--text); }
    .main .block-container { padding: 2.1rem 3.2rem 2.8rem; max-width: 1200px; }
    h1, h2, h3 { color: var(--text); letter-spacing: -0.02em; }
        h1 { font-size: 2.1rem; font-weight: 600; margin-bottom: 0.35rem; }
        h2 { font-size: 1.15rem; font-weight: 600; margin: 0 0 0.75rem; }
        p, label, .stMarkdown, .stText, .stCaption { color: var(--text); }
        .stSelectbox [data-baseweb="select"], .stSelectbox [data-baseweb="select"] * { color: var(--text); }
        .stFileUploader label { color: var(--text); }
        .stFileUploader div[data-testid="stFileUploaderDropzone"] { color: var(--text); background: var(--card); }
        .stFileUploader button { color: var(--text) !important; background: var(--card) !important; border: 1px solid var(--border) !important; }
        .stSelectbox > div[data-baseweb="select"] > div { background: var(--card); border-color: var(--border); border-radius: 10px; }
        .stFileUploader { border: 1.5px dashed var(--border); background: var(--card); border-radius: 12px; padding: 1rem; }
        .stImage { border-radius: 14px; overflow: hidden; box-shadow: 0 10px 28px rgba(0,0,0,0.06); border: 1px solid var(--border); }

        .hero { display: flex; justify-content: space-between; gap: 1.5rem; align-items: flex-end; margin-bottom: 1.6rem; }
        .hero p { color: var(--text-muted); margin: 0; max-width: 640px; }
        .badge { display: inline-block; padding: 0.3em 0.6em; font-size: 0.75rem; font-weight: 600; color: var(--text-muted); background: #EFE6DA; border-radius: 999px; margin-right: 0.4rem; }

    .panel { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 1.15rem; box-shadow: 0 8px 20px rgba(0,0,0,0.05); }
    .panel-controls { box-shadow: 0 6px 16px rgba(0,0,0,0.04); }
    .panel-preview { padding: 1.25rem; box-shadow: 0 16px 36px rgba(0,0,0,0.08); }
    .panel-title { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.6rem; }
        .panel-title h3 { font-size: 1.05rem; margin: 0; }
        .panel-subtitle { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.2rem; }
        .subtle { color: var(--text-muted); font-size: 0.9rem; }

    .control-stack { display: grid; gap: 0.85rem; margin-top: 0.4rem; }
    .control-actions { margin-top: 0.2rem; }
    .control-actions .stButton>button { width: 100%; }

    .stButton>button { border-radius: 12px; padding: 0 1.4rem; height: 44px; font-weight: 600; line-height: 1; display: inline-flex; align-items: center; justify-content: center; transition: all 0.2s ease; }
    .stButton>button:hover { transform: translateY(-1px); }
    .stButton>button:disabled { opacity: 0.6; box-shadow: none; transform: none; }
    .stButton>button[data-testid="baseButton-primary"] { background: var(--accent); color: #FFFDF9; border: none; box-shadow: 0 6px 14px rgba(200,107,54,0.25); }
    .stButton>button[data-testid="baseButton-primary"]:hover { background: var(--accent-hover); }
    .stButton>button[data-testid="baseButton-secondary"] { background: transparent; color: var(--text); border: 1px solid var(--border); box-shadow: none; }

    .preview-frame { background: #F9F4EC; border: 1px solid var(--border); border-radius: 18px; padding: 1.25rem; min-height: 0; display: block; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6); }
    .preview-frame img { max-width: 100%; height: auto; border-radius: 14px; border: 1px solid var(--border); box-shadow: 0 10px 24px rgba(0,0,0,0.06); }
    .preview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.85rem; }
    .preview-item { display: flex; flex-direction: column; gap: 0.45rem; }
    .preview-caption { font-size: 0.8rem; color: var(--text-muted); }
        .empty { color: var(--text-muted); font-size: 0.95rem; text-align: center; }

        .notice { padding: 0.45rem 0.7rem; border-radius: 10px; font-size: 0.85rem; border: 1px solid var(--border); background: #FFF8F1; color: var(--text); display: inline-flex; align-items: center; gap: 0.35rem; }
        .notice.success { border-color: rgba(102,122,62,0.4); background: #F6F8F1; color: var(--success); }
        .notice.error { border-color: rgba(177,78,58,0.4); background: #FDF3F1; color: var(--error); }

        .stats { display: flex; gap: 0.6rem; flex-wrap: wrap; }
        .stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.9rem; margin-top: 1rem; }
        .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem; box-shadow: 0 10px 24px rgba(0,0,0,0.05); }
        .stat-title { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.35rem; }
        .stat-value { font-size: 1.4rem; font-weight: 600; margin: 0; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
        .chip { border: 1px solid var(--border); padding: 0.35rem 0.65rem; border-radius: 999px; font-size: 0.85rem; background: #FFF9F3; color: var(--text); }

        .status-badge { padding: 0.25rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; background: #F1E9DD; color: var(--text-muted); border: 1px solid var(--border); }
    .download-row { display: grid; gap: 0.6rem; margin-top: 0.75rem; }
    .download-row .stButton>button { height: 42px; border-radius: 12px; background: #F1E9DD; color: var(--text); border: 1px solid var(--border); box-shadow: none; }
    .download-row .stButton>button:hover { background: #E7DDCF; transform: none; }

        @media (max-width: 900px) {
            .main .block-container { padding: 1.5rem 1.25rem 2rem; }
            .hero { flex-direction: column; align-items: flex-start; }
            .stat-grid { grid-template-columns: 1fr; }
        }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_yolo_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")
    return YOLO(str(model_path))

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def filter_reliable_boxes(result, image_size, min_conf: float, min_area_ratio: float):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Optional: explicitly filter for vehicles if using a generic COCO model
    # COCO vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    # We will just ensure confidence is high enough to rule out weak non-vehicle detections
    conf_mask = boxes.conf >= min_conf
    if not conf_mask.any():
        return None
    boxes = boxes[conf_mask]

    if min_area_ratio > 0:
        image_width, image_height = image_size
        image_area = float(image_width * image_height)
        if image_area > 0:
            xyxy = boxes.xyxy
            widths = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
            heights = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
            areas = widths * heights
            area_mask = areas >= (min_area_ratio * image_area)
            if not area_mask.any():
                return None
            boxes = boxes[area_mask]

    return boxes

# --- STATE ---
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'active_model' not in st.session_state:
    st.session_state.active_model = None

# --- UI ---
load_css()

st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.markdown(
    "<div>"
    "<span class='badge'>Vehicle Detection</span>"
    "<span class='badge'>YOLOv8</span>"
    "<h1>Vehicle Detection Demo</h1>"
    "<p>Upload an image, choose a model, and review detections in a refined preview.</p>"
    "<p class='subtle'>Best results on traffic or street-scene images.</p>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2.7], gap="large")

with col1:
    st.markdown("<div class='panel panel-controls'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'><div><h3>Controls</h3><div class='panel-subtitle'>Model + Image</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='control-stack'>", unsafe_allow_html=True)

    if not MODEL_FILES:
        st.markdown("<div class='notice error'>Không tìm thấy model trong thư mục models/.</div>", unsafe_allow_html=True)
        selected_model = None
    else:
        selected_model = st.selectbox(
            "**Model**",
            MODEL_FILES,
            index=0,
        )

    uploaded_files = st.file_uploader(
        "**Image**",
        type=['jpg', 'jpeg', 'png'],
        help="Only JPG/JPEG/PNG are supported.",
        accept_multiple_files=True
    )

    if uploaded_files:
        images = []
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                image = Image.open(uploaded_file).convert("RGB")
                images.append({"name": uploaded_file.name, "image": image})
            except UnidentifiedImageError:
                st.error(f"Ảnh {uploaded_file.name} không được hỗ trợ. Vui lòng dùng JPG/JPEG/PNG.")
        st.session_state.uploaded_images = images

    st.markdown("<div class='control-actions'>", unsafe_allow_html=True)
    button_col1, button_col2 = st.columns(2)

    detect_button = button_col1.button(
        "Detect",
        use_container_width=True,
        type="primary",
        disabled=(not st.session_state.uploaded_images or selected_model is None)
    )

    def reset_state():
        st.session_state.detection_results = None
        st.session_state.uploaded_images = []
        st.experimental_rerun()

    with button_col2:
        st.button("Reset", use_container_width=True, on_click=reset_state, type="secondary")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel panel-preview'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title'>"
        "<div><h3>Detection Output</h3><div class='panel-subtitle'>" +
        (selected_model if selected_model else "Select a model") +
        "</div></div>"
        "<span class='status-badge'>" +
        ("Detected" if st.session_state.detection_results and any(len(item["data"]) for item in st.session_state.detection_results)
         else "No results" if st.session_state.detection_results else "Ready") +
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )
    if st.session_state.detection_results:
        items = []
        for item in st.session_state.detection_results:
            image_b64 = image_to_base64(item['image'])
            items.append(
                "<div class='preview-item'>"
                f"<img src='data:image/png;base64,{image_b64}' alt='Detection Result' />"
                f"<div class='preview-caption'>{item['name']}</div>"
                "</div>"
            )
        has_reliable = any(len(item["data"]) for item in st.session_state.detection_results)
        notice_text = (
            "Detection complete" if has_reliable
            else "No reliable vehicle detected. Please try a traffic or street-scene image."
        )
        notice_class = "notice success" if has_reliable else "notice error"
        preview_html = (
            "<div class='preview-frame'>"
            "<div class='preview-grid'>"
            + "".join(items) +
            "</div>"
            f"<div class='{notice_class}' style='margin-top: 1rem;'>{notice_text}</div>"
            "</div>"
        )
        st.markdown(preview_html, unsafe_allow_html=True)
    elif st.session_state.uploaded_images:
        items = []
        for item in st.session_state.uploaded_images:
            image_b64 = image_to_base64(item['image'])
            items.append(
                "<div class='preview-item'>"
                f"<img src='data:image/png;base64,{image_b64}' alt='Original Image' />"
                f"<div class='preview-caption'>{item['name']}</div>"
                "</div>"
            )
        preview_html = (
            "<div class='preview-frame'>"
            "<div class='preview-grid'>"
            + "".join(items) +
            "</div>"
            "<div class='notice'>Ready to detect</div>"
            "</div>"
        )
        st.markdown(preview_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='preview-frame'><div class='empty'>Upload a JPG/PNG image to preview detection results.</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

if detect_button:
    if selected_model and st.session_state.uploaded_images:
        model_path = Path(MODELS_DIR) / selected_model
        try:
            model = load_yolo_model(model_path)
            with st.spinner('Running detection...'):
                input_images = [item["image"] for item in st.session_state.uploaded_images]
                results = model(input_images, conf=CONFIDENCE_THRESHOLD)
                packaged_results = []

                for item, result in zip(st.session_state.uploaded_images, results):
                    original_image = item["image"]
                    boxes = filter_reliable_boxes(
                        result,
                        original_image.size,
                        DISPLAY_CONFIDENCE_THRESHOLD,
                        MIN_BOX_AREA_RATIO
                    )

                    if boxes is not None and len(boxes) > 0:
                        result.boxes = boxes
                        result_image = Image.fromarray(
                            result.plot(line_width=2, font_size=0.6, conf=False)[:, :, ::-1]
                        )
                    else:
                        result_image = original_image.copy()

                    detected_objects = []
                    txt_lines = []
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            class_name = model.names[int(box.cls)]
                            confidence = float(box.conf)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detected_objects.append({
                                "Class": class_name,
                                "Confidence": f"{confidence:.2%}"
                            })
                            txt_lines.append(
                                f"{class_name} {confidence:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                            )

                    packaged_results.append({
                        "name": item["name"],
                        "image": result_image,
                        "original": original_image,
                        "data": detected_objects,
                        "txt": "\n".join(txt_lines) if txt_lines else ""
                    })

                st.session_state.detection_results = packaged_results
                st.session_state.active_model = selected_model
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Không thể load model: {e}")
            st.exception(e)
            st.stop()

if st.session_state.detection_results:
    results_data = []
    for result in st.session_state.detection_results:
        results_data.extend(result['data'])

    # Global Stats
    if not results_data:
        st.markdown(
            "<div class='notice error' style='margin-top: 1rem; width: 100%; justify-content: center; padding: 1rem; font-size: 1rem;'>"
            "No reliable vehicle detected in any of the images."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        class_counts = pd.DataFrame(results_data)['Class'].value_counts().reset_index()
        class_counts.columns = ['Vehicle Type', 'Count']

        st.markdown("<div class='stat-grid'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='stat-card'><div class='stat-title'>Total Objects</div><div class='stat-value'>{len(results_data)}</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='stat-card'><div class='stat-title'>Classes Found</div><div class='stat-value'>{class_counts.shape[0]}</div></div>",
            unsafe_allow_html=True
        )
        active_model_name = st.session_state.active_model or "-"
        st.markdown(
            f"<div class='stat-card'><div class='stat-title'>Active Model</div><div class='stat-value'>{active_model_name}</div></div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        chips = "".join(
            [f"<span class='chip'>{row['Vehicle Type']} · {row['Count']}</span>" for _, row in class_counts.iterrows()]
        )
        st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Batch Results Breakdown")
    
    combined_txt_lines = []
    
    for result in st.session_state.detection_results:
        img_name = result['name']
        num_dets = len(result['data'])
        
        # UI Rendering
        st.markdown(f"<h4 style='margin-top: 1.5rem;'>📄 {img_name}</h4>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1.2], gap="large")
        
        with colA:
            st.image(result['image'], use_container_width=True)
        
        with colB:
            st.write(f"**Total detections:** {num_dets}")
            if num_dets == 0:
                st.info("No reliable vehicle detected")
            else:
                for obj in result['data']:
                    st.write(f"- {obj['Class']} | {obj['Confidence']}")

        # TXT Report building
        combined_txt_lines.append(f"Image: {img_name}")
        combined_txt_lines.append(f"Total detections: {num_dets}")
        combined_txt_lines.append("")
        
        if num_dets == 0:
            combined_txt_lines.append("* No reliable vehicle detected")
        else:
            for obj in result['data']:
                conf_str = obj['Confidence']
                # Convert "82.50%" -> 0.82 or keep original if conversion fails
                try:
                    if str(conf_str).endswith('%'):
                        val_float = float(conf_str.replace('%', '')) / 100.0
                        combined_txt_lines.append(f"* {obj['Class']} | {val_float:.2f}")
                    else:
                        combined_txt_lines.append(f"* {obj['Class']} | {conf_str}")
                except Exception:
                    combined_txt_lines.append(f"* {obj['Class']} | {conf_str}")
                    
        combined_txt_lines.append("")
        if result != st.session_state.detection_results[-1]:
            combined_txt_lines.append("-" * 40)
            combined_txt_lines.append("")

    st.markdown("---")
    st.subheader("Export")
    combined_txt_out = "\n".join(combined_txt_lines)
    
    st.download_button(
        label="📄 Download TXT Report",
        data=combined_txt_out,
        file_name="batch_detection_report.txt",
        mime="text/plain",
        type="primary"
    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6E6258; font-size: 0.9rem;'>"
    "Built with Streamlit & YOLOv8. Best on traffic/road scenes."
    "</div>",
    unsafe_allow_html=True
)
