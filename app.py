import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, tempfile, os, random

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="✍️ Text → Handwriting",
    page_icon="✍️",
    layout="centered",
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f5f0e6;
    }
    .main {
        background-color: #fdfcf7;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #8B5E3C !important;
        color: white !important;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #6b4428 !important;
        color: #fff !important;
    }
    .stSidebar {
        background-color: #f7f3e9;
        border-right: 2px solid #e0d6c2;
    }
    .stTextArea textarea {
        background-color: #fffaf0 !important;
        border-radius: 10px !important;
        border: 1px solid #d1c7b7 !important;
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# App Title
# ----------------------------
st.title("✍️ Text → Handwriting Generator")
st.markdown(
    "Turn your digital text into realistic handwriting on a **notebook-style page** 📖"
)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Settings")

PAGE_SIZE = (1240, 1754)  # A4 ~150 DPI

font_upload = st.sidebar.file_uploader("Upload font (.ttf / .otf)", type=["ttf", "otf"])
font_size = st.sidebar.slider("Font size", 20, 120, 48)
line_spacing = st.sidebar.slider("Line spacing", 1.0, 2.0, 1.3, step=0.1)

default_colors = {
    "Black Ink": "#000000",
    "Coffee Ink ☕": "#4B2E2E",
    "Royal Blue": "#1A3E8A",
}
ink_choice = st.sidebar.selectbox("Ink Color", list(default_colors.keys()))
ink_color = default_colors[ink_choice]

margin = st.sidebar.slider("Page margin", 20, 150, 60)

# Extra Features
add_guidelines = st.sidebar.checkbox("Add notebook lines", value=True)
add_jitter = st.sidebar.checkbox("Add handwriting jitter", value=True)

bg_upload = st.sidebar.file_uploader("Background image (optional)", type=["png", "jpg", "jpeg"])

# ----------------------------
# Text Input
# ----------------------------
st.markdown("### ✏️ Write Your Text")
text = st.text_area("Input", height=300, placeholder="Start typing here...")

# ----------------------------
# Font Loader
# ----------------------------
def load_font(uploaded, size):
    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            return ImageFont.truetype(tmp.name, size=size)
    return ImageFont.load_default()

# ----------------------------
# Text Wrapping
# ----------------------------
def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = cur + " " + w if cur else w
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ----------------------------
# Page Rendering
# ----------------------------
def render_page(text, font, page_size, margin, spacing, ink_color, bg_image, jitter, guidelines):
    W, H = page_size
    # Background
    page = Image.new("RGB", (W, H), "#fdfcf7")
    if bg_image:
        page = bg_image.convert("RGB").resize((W, H))

    draw = ImageDraw.Draw(page)
    line_h = int(font.getbbox("Ay")[3] * spacing)
    x, y = margin, margin
    max_w = W - 2 * margin

    # Notebook lines
    if guidelines:
        for gy in range(margin, H - margin, line_h):
            draw.line((margin, gy, W - margin, gy), fill="#d6cfc2", width=2)

    lines = wrap_text(draw, text, font, max_w)

    # Draw text
    for line in lines:
        dx = random.randint(-2, 2) if jitter else 0
        dy = random.randint(-2, 2) if jitter else 0
        draw.text((x + dx, y + dy), line, font=font, fill=ink_color)
        y += line_h
        if y > H - margin:
            break

    return page

# ----------------------------
# Generate Button
# ----------------------------
if st.button("✨ Generate Handwriting"):
    if not text.strip():
        st.error("⚠️ Please enter some text")
    else:
        font = load_font(font_upload, font_size)
        bg_img = Image.open(bg_upload) if bg_upload else None

        page = render_page(
            text,
            font,
            PAGE_SIZE,
            margin,
            line_spacing,
            ink_color,
            bg_img,
            add_jitter,
            add_guidelines,
        )

        st.image(page, caption="📝 Preview", use_column_width=True)

        # PNG Download
        buf_png = io.BytesIO()
        page.save(buf_png, format="PNG")
        st.download_button(
            "⬇️ Download PNG",
            buf_png.getvalue(),
            "handwriting.png",
            "image/png",
        )

        # PDF Download
        buf_pdf = io.BytesIO()
        page.save(buf_pdf, format="PDF")
        st.download_button(
            "⬇️ Download PDF",
            buf_pdf.getvalue(),
            "handwriting.pdf",
            "application/pdf",
        )
