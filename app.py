# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import io, tempfile, os, random, math, textwrap

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="✍️ Text → Handwriting (Advanced)", page_icon="✍️", layout="wide")
st.title("✍️ Text → Handwriting Generator — Advanced")
st.markdown(
    "Generate messy, multi-page handwriting-style pages. Use responsibly — **no forgery/impersonation**."
)

# ----------------------------
# Helper: A4 and sizes
# ----------------------------
PAPER_SIZES = {
    "A4": (210, 297),       # mm
    "Letter": (215.9, 279.4),
}

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Paper / Output")
paper_choice = st.sidebar.selectbox("Paper size", list(PAPER_SIZES.keys()), index=0)
dpi = st.sidebar.selectbox("DPI (quality)", [150, 200, 300, 600], index=2)  # default 300
W_mm, H_mm = PAPER_SIZES[paper_choice]
PAGE_SIZE = (mm_to_px(W_mm, dpi), mm_to_px(H_mm, dpi))

st.sidebar.header("Handwriting Style")
style_choice = st.sidebar.selectbox("Style preset", ["Neat Cursive", "Messy Print", "Scribble (very messy)"])

st.sidebar.header("Font & Size")
font_upload = st.sidebar.file_uploader("Upload font (.ttf/.otf) — optional", type=["ttf", "otf"])
font_size = st.sidebar.slider("Base font size (pt)", 18, 220, 48)

st.sidebar.header("Ink & Paper")
ink_color = st.sidebar.color_picker("Ink color", "#151515")
paper_color = st.sidebar.color_picker("Paper color", "#fdfcf7")
add_notebook_lines = st.sidebar.checkbox("Add faint notebook lines", value=True)
margin_mm = st.sidebar.slider("Page margin (mm)", 5, 40, 18)
margin = mm_to_px(margin_mm, dpi)

st.sidebar.header("Realism controls")
jitter_amount = st.sidebar.slider("Per-character jitter (px)", 0, 20, 4)
rotation_amount = st.sidebar.slider("Per-character rotation (deg)", 0, 12, 4)
baseline_wobble = st.sidebar.slider("Baseline wobble (px)", 0, 12, 3)
ink_spread = st.sidebar.slider("Ink spread / bleed (px)", 0, 8, 2)
pressure_variation = st.sidebar.slider("Pressure variation (0-1)", 0.0, 1.0, 0.35)
smudge_amount = st.sidebar.slider("Smudge intensity (0-1)", 0.0, 1.0, 0.12)

st.sidebar.header("Layout")
line_spacing = st.sidebar.slider("Line spacing multiplier", 1.0, 2.0, 1.35, step=0.05)
indent_first_line = st.sidebar.checkbox("Indent paragraphs", value=True)

st.sidebar.header("Output")
preview_scale = st.sidebar.slider("Preview scale (for big DPI)", 0.25, 1.0, 0.5)
generate_btn = st.sidebar.button("✨ Generate Handwriting")

# ----------------------------
# Text input
# ----------------------------
st.markdown("### ✏️ Text to handwrite")
input_text = st.text_area("Paste or type the text here", height=320, placeholder="Start typing...")

# ----------------------------
# Font loader (supports upload; fallback tries common fonts)
# ----------------------------
def load_font(uploaded_file, size):
    # If user uploaded font, use it
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            return ImageFont.truetype(tmp.name, size=size)
    # Try some common system fonts
    candidates = ["DejaVuSans.ttf", "Arial.ttf", "LiberationSerif-Regular.ttf"]
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size=size)
        except Exception:
            pass
    # fallback
    return ImageFont.load_default()

# ----------------------------
# Style presets (tweak many parameters per style)
# ----------------------------
STYLE_PRESETS = {
    "Neat Cursive": dict(
        char_spacing_variation=0.6,
        char_rotation_factor=0.6,
        smear_probability=0.04,
        double_stroke=0.2,
    ),
    "Messy Print": dict(
        char_spacing_variation=1.0,
        char_rotation_factor=1.0,
        smear_probability=0.12,
        double_stroke=0.35,
    ),
    "Scribble (very messy)": dict(
        char_spacing_variation=1.6,
        char_rotation_factor=1.5,
        smear_probability=0.28,
        double_stroke=0.6,
    ),
}

preset = STYLE_PRESETS[style_choice]

# ----------------------------
# Low-level helpers: create char image and paste with transforms
# ----------------------------
def render_char_image(ch, font, ink_color, pressure, stroke_width=1):
    # Render a single character as RGBA so we can rotate/transform it.
    # Pressure affects alpha and stroke-like thickness (we simulate by drawing multiple times).
    mask = Image.new("RGBA", font.getmask(ch).size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    # simulate thicker ink by drawing multiple slightly offset glyphs
    alpha = int(255 * (0.6 + 0.4 * pressure))
    times = 1 + int(round(stroke_width * pressure))
    w, h = mask.size
    cx = w // 2
    cy = h // 2
    for i in range(times):
        ox = random.randint(-1, 1)
        oy = random.randint(-1, 1)
        draw.text((ox, oy), ch, font=font, fill=hex_to_rgba(ink_color, alpha))
    return mask

def hex_to_rgba(hexcol, alpha=255):
    hexcol = hexcol.lstrip("#")
    lv = len(hexcol)
    if lv == 3:
        r,g,b = [int(hexcol[i]*2, 16) for i in range(3)]
    else:
        r,g,b = int(hexcol[0:2],16), int(hexcol[2:4],16), int(hexcol[4:6],16)
    return (r,g,b,int(alpha))

# ----------------------------
# Effects: ink bleed, smudge, noise
# ----------------------------
def add_ink_spread(page, spread_px):
    if spread_px <= 0:
        return page
    # simple spread: blur and multiply onto original
    spread = page.filter(ImageFilter.GaussianBlur(radius=spread_px))
    return Image.blend(page, spread, alpha=0.35)

def add_smudges(page, amount=0.1):
    if amount <= 0:
        return page
    W,H = page.size
    sm = Image.new("RGBA", (W,H), (0,0,0,0))
    draw = ImageDraw.Draw(sm)
    # add few random gaussian blobs of paper-color-ish blurred
    for _ in range(int(3 + amount * 12)):
        rx = random.randint(0, W)
        ry = random.randint(0, H)
        r = int(min(W,H) * (0.02 + random.random() * 0.08 * amount))
        color = (0,0,0,int(18 + random.random()*40*amount))
        draw.ellipse((rx-r, ry-r, rx+r, ry+r), fill=color)
    sm = sm.filter(ImageFilter.GaussianBlur(radius=6 * amount))
    page = Image.alpha_composite(page.convert("RGBA"), sm).convert("RGB")
    return page

def add_paper_texture(page, intensity=0.06):
    if intensity <= 0:
        return page
    arr = np.asarray(page).astype(np.float32) / 255.0
    W,H,_ = arr.shape
    noise = (np.random.randn(W,H) * intensity).reshape(W,H,1)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

# ----------------------------
# Main renderer: generate multiple pages
# ----------------------------
def generate_pages(text, font, page_size, margin, line_spacing, ink_color, paper_color,
                   jitter, rotation_deg, baseline_wobble, ink_spread_px, smudge_amt,
                   preset, indent_first):
    W,H = page_size
    pages = []
    words = text.replace("\r\n", "\n").split("\n")
    # We'll create a drawing surface and place text line-by-line with per-character placement.
    cur_page = Image.new("RGB", (W,H), paper_color)
    page_draw = ImageDraw.Draw(cur_page)
    y = margin
    max_w = W - 2*margin
    # approximate line height using 'Hg' to include ascenders/descenders
    sample_height = font.getbbox("Hg")[3]
    line_h = int(sample_height * line_spacing)

    def start_new_page():
        nonlocal cur_page, page_draw, y
        pages.append(cur_page)
        cur_page = Image.new("RGB", (W,H), paper_color)
        page_draw = ImageDraw.Draw(cur_page)
        y = margin

    # draw optional notebook faint lines
    if add_notebook_lines:
        for gy in range(margin, H - margin, line_h):
            page_draw.line((margin, gy, W - margin, gy), fill=(214,207,194), width=max(1, int(dpi/300)))

    for paragraph in words:
        if indent_first and paragraph.strip():
            paragraph = "    " + paragraph.strip()
        # wrap paragraph to max_w using draw.textbbox measurement
        # We'll split into words and greedily place them, generating per-character placements.
        words_in_para = paragraph.split(" ")
        x = margin
        # small paragraph top gap
        if y + line_h > H - margin:
            start_new_page()
        # handle explicit empty lines as paragraph breaks
        if paragraph.strip() == "":
            y += int(line_h * 0.5)
            continue

        for widx, word in enumerate(words_in_para):
            # measure word width (rough)
            w_bbox = page_draw.textbbox((0,0), word + " ", font=font)
            word_width = w_bbox[2] - w_bbox[0]
            if x + word_width > W - margin:
                # move to next line
                x = margin
                y += line_h
                # new page if needed
                if y + line_h > H - margin:
                    start_new_page()
            # render each character in the word
            for c in word + (" " if widx < len(words_in_para)-1 else ""):
                # pressure
                pressure = max(0.15, min(1.0, random.random() * pressure_variation + (1.0 - pressure_variation)))
                # stroke width base
                stroke_w = 1.0 + ink_spread_px * 0.6
                ch_img = render_char_image(c, font, ink_color, pressure, stroke_width=stroke_w)
                cw, ch = ch_img.size
                # random jitter and rotation
                jx = int(round(random.uniform(-jitter, jitter) * preset["char_spacing_variation"]))
                jy = int(round(random.uniform(-baseline_wobble, baseline_wobble)))
                rot = random.uniform(-rotation_deg, rotation_deg) * preset["char_rotation_factor"]
                # apply rotation
                ch_img = ch_img.rotate(rot, resample=Image.BICUBIC, expand=True)
                # optionally double stroke (draw character again lightly to simulate shaky stroke)
                if random.random() < preset["double_stroke"]:
                    extra = ch_img.copy().filter(ImageFilter.GaussianBlur(radius=0.5))
                    ch_img = Image.alpha_composite(Image.new("RGBA", extra.size, (0,0,0,0)), extra)
                # paste
                px = x + jx
                py = y + jy
                # If char would go outside right edge, wrap line
                if px + ch_img.size[0] > W - margin:
                    x = margin
                    y += line_h
                    px = x
                    py = y + jy
                    if y + line_h > H - margin:
                        start_new_page()
                        px = x
                        py = y + jy
                # paste with alpha
                cur_page.paste(ch_img, (px, py), ch_img)
                # advance x based on measured cw but vary spacing
                advance = int((cw * (0.8 + random.uniform(-0.12, 0.12) * preset["char_spacing_variation"])))
                x += max(advance, 1)
            # small addition between words
        # paragraph gap
        y += line_h

    # append last page
    pages.append(cur_page)

    # Post-process pages: ink spread, smudges, optional texture
    processed = []
    for p in pages:
        p = add_ink_spread(p, ink_spread_px)
        p = add_smudges(p, smudge_amt)
        p = add_paper_texture(p, intensity=0.02)
        processed.append(p)
    return processed

# ----------------------------
# Utility: save pages to multi-page PDF bytes
# ----------------------------
def pages_to_pdf_bytes(pages):
    buf = io.BytesIO()
    rgb_pages = [p.convert("RGB") for p in pages]
    if len(rgb_pages) == 1:
        rgb_pages[0].save(buf, format="PDF")
    else:
        rgb_pages[0].save(buf, format="PDF", save_all=True, append_images=rgb_pages[1:])
    return buf.getvalue()

# ----------------------------
# UI: Generate
# ----------------------------
if generate_btn:
    if not input_text.strip():
        st.error("⚠️ Please enter some text first.")
    else:
        with st.spinner("Rendering pages... this may take a few seconds for high DPI / long text"):
            # Load font
            pil_font = load_font(font_upload, int(font_size * dpi/72))  # convert pt to px roughly
            # Generate pages
            pages = generate_pages(
                input_text,
                pil_font,
                PAGE_SIZE,
                margin,
                line_spacing,
                ink_color,
                paper_color,
                jitter_amount,
                rotation_amount,
                baseline_wobble,
                ink_spread,
                smudge_amount,
                preset,
                indent_first_line,
            )

            # Preview scaled
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Preview (scaled)")
                for i, p in enumerate(pages):
                    display = p.resize((int(p.size[0]*preview_scale), int(p.size[1]*preview_scale)), Image.LANCZOS)
                    st.image(display, caption=f"Page {i+1}", use_column_width=False)
            with col2:
                st.subheader("Download")
                # PNG downloads (first page only quick)
                png_buf = io.BytesIO()
                pages[0].save(png_buf, format="PNG")
                st.download_button("⬇️ Download Page 1 PNG", png_buf.getvalue(), "handwritten_page1.png", "image/png")
                # Single ZIP-like approach: create a multi-page PDF
                pdf_bytes = pages_to_pdf_bytes(pages)
                st.download_button("⬇️ Download multi-page PDF", pdf_bytes, "handwritten_pages.pdf", "application/pdf")
                # Offer all PNGs in-memory as a zip? For simplicity, return multiple buttons if <=5 pages
                if len(pages) <= 8:
                    for i, p in enumerate(pages):
                        buf = io.BytesIO()
                        p.save(buf, format="PNG")
                        st.download_button(f"PNG: Page {i+1}", buf.getvalue(), f"handwritten_page_{i+1}.png", "image/png")
                else:
                    st.info("For many pages, download the PDF above. If you need separate PNGs for each page, I can add zip export.")

        st.success(f"Rendered {len(pages)} page(s). Tweak the realism sliders to adjust messiness.")

# ----------------------------
# Tips & Notes
# ----------------------------
st.markdown("---")
st.markdown(
    """
    **Tips to increase realism**
    - Upload a handwriting-like `.ttf` font (many free handwriting fonts exist) — combine with the jitter and smudge controls.
    - Increase `ink spread` and `smudge` a small amount for more bleed and blur.
    - Use higher DPI (300 or 600) if you plan to print — note this increases generation time.
    - For truly indistinguishable handwriting you'd need real handwriting samples and trained models (GANs / diffusion), but this generator produces plausible, inconsistent handwriting for creative uses.
    
    **Ethics reminder:** Do **not** use this tool to forge signatures, impersonate someone, or misrepresent documents. Use for mockups, art, and practice only.
    """
)

# End of app.py
