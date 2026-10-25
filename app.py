
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "wheat_leaf_disease_cnn_improved.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)

DATASET_CANDIDATES = [
    BASE_DIR / "non_dup_dataset",
    Path(r"D:\non_dup_dataset"),
]

DATASET_DIR = next(
    (path for path in DATASET_CANDIDATES if path.is_dir()),
    None,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {
    "Brown rust": {
        "subtitle": "Reddish-brown pustules on leaf surfaces",
        "overview": (
            "Brown rust is a fungal disease of wheat. It commonly produces "
            "scattered reddish-brown or orange-brown pustules on leaves."
        ),
        "symptoms": [
            "Small reddish-brown pustules scattered across leaves.",
            "Pustules may release coloured spores when rubbed.",
            "Severe infection can reduce green leaf area and affect grain filling.",
        ],
        "actions": [
            "Inspect several plants instead of relying on one leaf.",
            "Record affected areas and monitor whether symptoms spread.",
            "Consider resistant varieties when planning future crops.",
            "Ask a local agricultural expert whether treatment is appropriate.",
        ],
        "treatment": (
            "Management depends on disease severity, crop stage, wheat variety "
            "and local conditions. Confirm the diagnosis before treatment."
        ),
    },
    "Yellow rust": {
        "subtitle": "Yellow or orange pustules, often arranged in stripes",
        "overview": (
            "Yellow rust is a fungal wheat disease that commonly produces "
            "yellow-orange pustules in stripes along the leaves."
        ),
        "symptoms": [
            "Yellow or orange powdery pustules.",
            "Pustules may form stripes along the leaf.",
            "Symptoms may spread when conditions favour the fungus.",
        ],
        "actions": [
            "Inspect leaves for yellow-orange stripes.",
            "Monitor the crop regularly, especially in cool, humid weather.",
            "Do not identify a disease from colour alone.",
            "Seek local agricultural advice if symptoms spread rapidly.",
        ],
        "treatment": (
            "Early assessment can help guide management. Follow local "
            "agricultural advice and product-label directions."
        ),
    },
    "Loose Smut": {
        "subtitle": "Dark spores affecting wheat heads",
        "overview": (
            "Loose smut primarily affects wheat heads. Infected heads may be "
            "replaced by masses of dark spores, so a leaf-only image may be "
            "insufficient for identification."
        ),
        "symptoms": [
            "Dark, powdery spore masses replacing grain structures.",
            "Infected heads may become visible as the crop emerges.",
            "Other head diseases or physical damage can look similar.",
        ],
        "actions": [
            "Inspect wheat heads as well as leaves.",
            "Record affected plants and consult an agricultural expert.",
            "Ask about certified seed and seed-health measures for future sowings.",
            "Discuss prevention options with a qualified professional.",
        ],
        "treatment": (
            "An infected head cannot be restored. Management often focuses "
            "on seed health and prevention in future crops."
        ),
    },
    "Septoria": {
        "subtitle": "Leaf lesions and spots",
        "overview": (
            "Septoria diseases can cause lesions on wheat leaves. These may "
            "be brown, tan or greyish, sometimes with darker specks."
        ),
        "symptoms": [
            "Brown, tan or greyish lesions on leaves.",
            "Spots may enlarge or merge.",
            "Dark specks may be visible within lesions.",
        ],
        "actions": [
            "Inspect several leaves and monitor whether lesions increase.",
            "Avoid relying on a single spot or colour for identification.",
            "Follow locally recommended crop and residue-management practices.",
            "Ask an agricultural expert to confirm the cause.",
        ],
        "treatment": (
            "Management depends on the cause, crop stage and severity. "
            "Protecting remaining healthy tissue requires correct assessment."
        ),
    },
    "Healthy": {
        "subtitle": "The model classified this image as Healthy",
        "overview": (
            "The model classified the image as Healthy based on patterns "
            "learned during training. This does not guarantee that the plant "
            "is free from disease, pests or nutrient deficiencies."
        ),
        "symptoms": [
            "The model did not rank a target disease class highest.",
            "Image quality and the visible plant part can affect the result.",
        ],
        "actions": [
            "Continue routine field inspections.",
            "Capture clear images of multiple leaves and plants.",
            "Check for unusual spots, yellowing or changes in growth.",
            "Seek expert assessment if the crop still appears unhealthy.",
        ],
        "treatment": (
            "Do not use this prediction alone to decide whether treatment "
            "is needed. Continue monitoring the crop."
        ),
    },
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WheatCare AI | Wheat Leaf Health",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING — IMPROVED TEXT CONTRAST
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --forest: #174c35;
        --green: #24734c;
        --cream: #f7f8f2;
        --ink: #20352a;
        --muted: #53675b;
        --line: #dce6dc;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--cream);
        color: var(--ink);
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--forest) !important;
        font-weight: 750 !important;
    }

    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: var(--ink);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #edf3eb;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--ink) !important;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(125deg, #123d2c, #24734c);
        padding: 2.2rem;
        border-radius: 22px;
        margin: 0.5rem 0 1.5rem;
        box-shadow: 0 10px 28px rgba(23, 76, 53, 0.12);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.2;
    }

    .hero p {
        color: #f0f8f1 !important;
        line-height: 1.8;
        max-width: 780px;
    }

    .eyebrow {
        color: #d4f0d9 !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 0.7rem;
    }

    /* Content cards */
    .panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(24, 60, 39, 0.04);
    }

    .panel h3 {
        color: var(--forest) !important;
    }

    .panel p {
        color: var(--ink) !important;
        line-height: 1.65;
    }

    .small-label {
        color: #53675b !important;
        font-size: 0.77rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-class {
        color: var(--forest) !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .tag {
        display: inline-block;
        background: #e6f1e7;
        color: #174c35 !important;
        border-radius: 25px;
        padding: 0.4rem 0.8rem;
        font-weight: 700;
    }

    /* Upload and camera areas: readable dark text on light background */
    [data-testid="stFileUploader"],
    [data-testid="stCameraInput"] {
        background: #ffffff !important;
        color: #20352a !important;
        border: 1px solid #cbd9cc !important;
        border-radius: 14px !important;
        padding: 0.65rem !important;
    }

    [data-testid="stFileUploader"] *,
    [data-testid="stCameraInput"] * {
        color: #20352a !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stCameraInput"] section,
    [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] {
        background: #f1f8f2 !important;
        color: #20352a !important;
        border-color: #a9c7b0 !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stCameraInput"] button {
        background: #e4f1e6 !important;
        color: #173f2c !important;
        border: 1px solid #a9c7b0 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stCameraInput"] button:hover {
        background: #c8e5cf !important;
        color: #123d2c !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stCameraInput"] p {
        color: #30483a !important;
    }

    [data-testid="stCameraInput"] a {
        color: #24734c !important;
        text-decoration: underline !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    /* Metrics */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #174c35 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: #20352a !important;
    }

    /* Buttons and download buttons: readable on light background */
    div.stButton > button,
    div.stDownloadButton > button {
        background: #e4f1e6 !important;
        color: #173f2c !important;
        border: 1px solid #a9c7b0 !important;
        border-radius: 10px !important;
        min-height: 2.7rem;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #cfe6d3 !important;
        color: #123d2c !important;
        border-color: #79a989 !important;
    }

    div.stButton > button[kind="primary"] {
        background: #216447 !important;
        color: #ffffff !important;
        border: 1px solid #216447 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #174c35 !important;
        color: #ffffff !important;
    }

    /* Data table */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        color: #20352a !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #53675b !important;
        font-size: 0.9rem;
        line-height: 1.8;
        padding: 2rem 0 1rem;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 1.25rem;
            border-radius: 15px;
        }

        .panel {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty list.")

    return names


# ============================================================
# DATASET SAMPLE IMAGE HELPERS
# ============================================================

def normalise_name(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


@st.cache_data
def find_sample_images(class_name, limit=3):
    if DATASET_DIR is None:
        return []

    target = normalise_name(class_name)
    matches = []

    try:
        for folder in DATASET_DIR.rglob("*"):
            if folder.is_dir() and normalise_name(folder.name) == target:
                matches = [
                    path for path in folder.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                if matches:
                    break
    except OSError:
        return []

    return [str(path) for path in sorted(matches)[:limit]]


def open_small_sample(image_path):
    with Image.open(image_path) as source:
        source = ImageOps.exif_transpose(source)
        source.thumbnail((600, 400), Image.Resampling.LANCZOS)
        image = source.convert("RGB").copy()

    return ImageOps.fit(
        image,
        (600, 360),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):
    model = load_model()
    class_names = load_class_names()

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(image_array, axis=0)

    raw_scores = np.asarray(
        model.predict(batch, verbose=0)[0]
    ).reshape(-1)

    if len(raw_scores) != len(class_names):
        raise ValueError(
            "Model output count does not match class_names.json."
        )

    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("The model returned invalid prediction values.")

    if (
        np.any(raw_scores < 0)
        or np.any(raw_scores > 1)
        or not np.isclose(raw_scores.sum(), 1.0, atol=0.02)
    ):
        raise ValueError(
            "Model outputs do not appear to be probabilities. "
            "Check the model's final layer."
        )

    order = np.argsort(raw_scores)[::-1]
    best_index = int(order[0])

    return {
        "predicted_class": class_names[best_index],
        "confidence": float(raw_scores[best_index]),
        "scores": [
            {
                "class": class_names[int(index)],
                "score": float(raw_scores[index]),
            }
            for index in order
        ],
    }


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


def go_to_detection():
    st.session_state.page = "Disease Detection"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🌾 WheatCare AI")
    st.caption("Wheat leaf health, made easier")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Disease Detection",
            "Disease Guide",
            "Prediction History",
            "About Project",
        ],
        key="page",
    )

    st.divider()
    st.markdown("### YOUR FIELD ASSISTANT")
    st.write("AI-based wheat image classification")
    st.caption("Python · CNN · TensorFlow · Streamlit")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">🌱 AGRICULTURAL INTELLIGENCE</div>
            <h1>Know your wheat.<br>Protect your harvest.</h1>
            <p>
                Explore a CNN-powered wheat image classification tool.
                Upload a photograph or capture a new image to see which
                of the five trained classes best matches it.
            </p>
            <p>
                A simple image-based screening tool for an academic project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your wheat leaf assistant")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="panel">
                <div class="small-label">01 · ANALYSE</div>
                <h3>📷 Capture or upload</h3>
                <p>Use your camera or choose a saved image.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="panel">
                <div class="small-label">02 · CLASSIFY</div>
                <h3>🧠 CNN prediction</h3>
                <p>View the predicted class and class-wise model scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="panel">
                <div class="small-label">03 · LEARN</div>
                <h3>🌾 Disease guide</h3>
                <p>Explore typical symptoms and general crop guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Ready to check an image?")
    st.write(
        "Start with a clear, well-lit photograph showing the relevant "
        "part of the wheat plant."
    )

    st.button(
        "🌿 Start Disease Detection",
        type="primary",
        use_container_width=True,
        on_click=go_to_detection,
    )

    st.info(
        "This model is a screening aid, not a confirmed diagnosis. "
        "A high model score does not guarantee a correct result."
    )


# ============================================================
# DISEASE DETECTION PAGE
# ============================================================

elif page == "Disease Detection":
    st.title("🌿 Disease Detection")
    st.write("Upload an image or use your camera to analyse a wheat plant.")

    input_method = st.radio(
        "How would you like to provide the image?",
        ["Upload an image", "Take a photo"],
        horizontal=True,
        key="image_source",
    )

    uploaded_file = None

    if input_method == "Upload an image":
        uploaded_file = st.file_uploader(
            "Choose a wheat image",
            type=["jpg", "jpeg", "png"],
            help="Use a sharp, well-lit image with the plant clearly visible.",
            key="leaf_upload",
        )
    else:
        uploaded_file = st.camera_input(
            "Capture a wheat image",
            key="leaf_camera",
        )

    if uploaded_file is not None:
        try:
            with Image.open(uploaded_file) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").copy()

            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("### Image preview")
                st.image(
                    image,
                    caption="Image ready for analysis",
                    use_container_width=True,
                )
                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

            with right:
                st.markdown("### Run CNN analysis")
                st.write(
                    "The image will be resized to 224 × 224 pixels "
                    "and normalized before prediction."
                )

                if st.button(
                    "Analyse Leaf Image",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Analysing image..."):
                            prediction = predict_image(image)

                        result = {
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "filename": getattr(
                                uploaded_file,
                                "name",
                                "Camera capture",
                            ),
                            **prediction,
                        }

                        st.session_state.latest_result = result
                        st.session_state.history.insert(0, result)
                        st.session_state.history = (
                            st.session_state.history[:100]
                        )

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

            result = st.session_state.latest_result

            if result is not None:
                st.divider()
                st.markdown("## 🌱 Prediction Result")
                result_left, result_right = st.columns(2)

                with result_left:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div class="small-label">MODEL PREDICTION</div>
                            <div class="result-class">
                                {result["predicted_class"]}
                            </div>
                            <p>
                                <span class="tag">
                                    {result["confidence"] * 100:.2f}%
                                    model score
                                </span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    details = DISEASE_INFO.get(
                        result["predicted_class"],
                        {
                            "subtitle": "Classification result",
                            "overview": "No additional information is available.",
                        },
                    )

                    st.markdown("### What this result means")
                    st.write(details["subtitle"])
                    st.write(details["overview"])

                with result_right:
                    st.markdown("### Class-wise scores")

                    for item in result["scores"]:
                        st.write(
                            f'**{item["class"]}** — '
                            f'{item["score"] * 100:.2f}%'
                        )
                        st.progress(
                            float(np.clip(item["score"], 0.0, 1.0))
                        )

                st.warning(
                    "Do not choose a crop treatment from the model score "
                    "alone. Confirm symptoms with an agricultural expert."
                )

                if st.button("Clear displayed result"):
                    st.session_state.latest_result = None
                    st.rerun()

        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not read the selected image: {error}")

    else:
        st.info("Choose an image or take a photo to begin.")


# ============================================================
# DISEASE GUIDE PAGE
# ============================================================

elif page == "Disease Guide":
    st.title("🌾 Wheat Disease Guide")
    st.write(
        "Explore common symptoms and general management information. "
        "Sample images are loaded from your local dataset when available."
    )

    if DATASET_DIR is None:
        st.info(
            "Dataset samples were not found automatically. Check that "
            "your dataset is in a folder named non_dup_dataset inside "
            "your project directory, or at D:\\non_dup_dataset."
        )
    else:
        st.caption(f"Dataset directory: {DATASET_DIR}")

    for disease_name, details in DISEASE_INFO.items():
        st.divider()
        st.subheader(disease_name)
        st.write(details["subtitle"])

        samples = find_sample_images(disease_name, limit=3)

        if samples:
            image_columns = st.columns(len(samples))

            for column, image_path in zip(image_columns, samples):
                try:
                    sample_image = open_small_sample(image_path)

                    with column:
                        st.image(
                            sample_image,
                            caption=f"{disease_name} sample",
                            use_container_width=True,
                        )

                except (OSError, ValueError, UnidentifiedImageError):
                    with column:
                        st.caption("This sample image could not be loaded.")
        else:
            st.caption("No matching sample image was found for this class.")

        st.markdown("### About and symptoms")
        st.write(details["overview"])

        st.markdown("### What to look for")
        for symptom in details["symptoms"]:
            st.markdown(f"- {symptom}")

        st.markdown("### Management and prevention")
        for number, action in enumerate(details["actions"], start=1):
            st.markdown(f"{number}. {action}")

        st.markdown("### Management notes")
        st.write(details["treatment"])

        with st.expander("When should I seek agricultural advice?"):
            st.write(
                "Seek advice when symptoms spread quickly, several plants "
                "are affected, or a treatment decision depends on the "
                "diagnosis. A local agricultural extension officer or "
                "qualified plant pathologist can assess the crop."
            )

    st.info(
        "These notes are educational, not a confirmed diagnosis or "
        "pesticide prescription. Follow local professional advice and "
        "the label of any locally registered crop-protection product."
    )


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":
    st.title("📊 Prediction History")
    st.write(
        "Review predictions recorded during the current Streamlit session."
    )

    if st.session_state.history:
        rows = [
            {
                "Time": item["time"],
                "Image": item["filename"],
                "Predicted class": item["predicted_class"],
                "Model score": f'{item["confidence"] * 100:.2f}%',
            }
            for item in st.session_state.history
        ]

        history_df = pd.DataFrame(rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="wheatcare_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.rerun()

    else:
        st.info(
            "No predictions recorded yet. Analyse an image to see "
            "your prediction history here."
        )

    st.caption(
        "History is stored in the current session only. It is not "
        "permanently saved to a database."
    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">🌾 OUR PROJECT</div>
            <h1>Technology for healthier wheat.</h1>
            <p>
                WheatCare AI is an academic project exploring how computer
                vision can classify wheat images and make model results
                easier to understand.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## From image to insight")
    step1, step2 = st.columns(2)

    with step1:
        st.markdown(
            """
            <div class="panel">
                <div class="small-label">STEP 01</div>
                <h3>📷 Capture</h3>
                <p>The user uploads an image or takes a photograph.</p>
            </div>
            <div class="panel">
                <div class="small-label">STEP 02</div>
                <h3>🧹 Preprocess</h3>
                <p>The image is converted to RGB, resized to 224 × 224
                pixels and normalized to the range 0–1.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="panel">
                <div class="small-label">STEP 03</div>
                <h3>🧠 Predict</h3>
                <p>The trained CNN produces scores for five saved classes.</p>
            </div>
            <div class="panel">
                <div class="small-label">STEP 04</div>
                <h3>📋 Understand</h3>
                <p>The application displays the predicted class, model
                score and general educational information.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Project at a glance")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Image size", "224 × 224")
    metric2.metric("Output classes", "5")
    metric3.metric("Core model", "CNN")

    st.markdown("## Technology stack")
    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown(
            '<div class="panel"><h3>Python</h3>'
            '<p>Application logic</p></div>',
            unsafe_allow_html=True,
        )

    with tech2:
        st.markdown(
            '<div class="panel"><h3>TensorFlow</h3>'
            '<p>Trained CNN model</p></div>',
            unsafe_allow_html=True,
        )

    with tech3:
        st.markdown(
            '<div class="panel"><h3>NumPy</h3>'
            '<p>Image processing arrays</p></div>',
            unsafe_allow_html=True,
        )

    with tech4:
        st.markdown(
            '<div class="panel"><h3>Streamlit</h3>'
            '<p>Web interface</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## System capabilities and limitations")
    can, cannot = st.columns(2)

    with can:
        st.success("Designed to help with")
        st.markdown(
            "- Classifying images into five trained classes.\n"
            "- Displaying class-wise model scores.\n"
            "- Showing educational disease information.\n"
            "- Demonstrating a CNN-based workflow in a web application."
        )

    with cannot:
        st.warning("Important limitations")
        st.markdown(
            "- It cannot guarantee a correct diagnosis.\n"
            "- A leaf image may not show the full plant condition.\n"
            "- Model scores do not measure disease severity.\n"
            "- It does not replace field inspection or expert advice."
        )

    st.markdown("## Future improvements")
    st.write(
        "Possible extensions include persistent prediction history, "
        "database integration, multilingual support, mobile-friendly "
        "access and evaluation using independent field photographs."
    )

    st.info(
        "WheatCare AI is an academic prototype. Its performance in "
        "real farming conditions requires independent evaluation."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌾 WheatCare AI<br>
        CNN-based wheat image classification<br>
        Academic project prototype · Verify results before treatment
    </div>
    """,
    unsafe_allow_html=True,
)
