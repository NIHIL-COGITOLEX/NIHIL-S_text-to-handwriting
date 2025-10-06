"""
app.py — Advanced Text → Handwriting (Streamlit)

Features
- A4 output at configurable DPI (default 300)
- Real notebook page (upload background image or draw ruled lines + margin)
- Multi-page output (automatically paginates)
- Realistic messy cursive style with many realism tunables
- PNG per-page and a multi-page PDF export
- Well documented and segmented for readability

Usage
- Place this file in your GitHub repo and deploy to Streamlit Cloud
- Optionally upload a handwriting-like TTF/OTF and/or a paper background image
"""

import os
import io
import math
import tempfile
import random
import textwrap
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# ----------------------------
# App metadata & page config
# ----------------------------
st.set_page_config(page_title="✍️ Text → Handwriting — Advanced", page_icon="✍️", layout="wide")
st.title("✍️ Text → Handwriting — Advanced (Streamlit)")
st.markdown(
    """
Generate high-quality A4 handwriting pages with realistic inconsistency.
**Use ethically** — do not forge or impersonate.
"""
)

# ----------------------------
# Constants & helpers
# ----------------------------
PAPER_SIZES_MM = {
    "A4": (210.0, 297.0),
    "Letter": (216.0, 279.0),
}


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimeters to pixels for a given DPI."""
    return int(round(mm / 25.4 * dpi))


def pt_to_px(pt: float, dpi: int) -> int:
    """Convert points to pixels for a given DPI. (1pt = 1/72 in)"""
    return int(round(pt * dpi / 72.0))


def hex_to_rgba_tuple(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Convert `#RRGGBB` or `RRGGBB` to an (r,g,b,a) tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b, alpha)


def safe_truncate_text(t: str, max_chars=1000000) -> str:
    """Guard against extremely huge inputs causing memory issues."""
    if len(t) > max_chars:
        return t[:max_chars] + "\n\n[TRUNCATED]"
    return t


# ----------------------------
# Sidebar: user controls
# ----------------------------
st.sidebar.header("Paper / Output settings")
paper_choice = st.sidebar.selectbox("Paper size", list(PAPER_SIZES_MM.keys()), index=0)
dpi = st.sidebar.selectbox("DPI (print quality)", [150, 200, 300, 600], index=2)
use_paper_bg = st.sidebar.checkbox("Use uploaded paper background image (recommended)", value=True)
paper_bg_upload = st.sidebar.file_uploader("Upload paper background image (png/jpg) — optional", type=["png", "jpg", "jpeg"])

st.sidebar.header("Handwriting style")
style_choice = st.sidebar.selectbox("Handwriting style", ["Neat Cursive (slightly messy)", "Messy Cursive", "Natural Scribble"])
# We'll map these to internal presets below

st.sidebar.header("Font & size")
font_upload = st.sidebar.file_uploader("Upload handwriting font (.ttf/.otf) — optional", type=["ttf", "otf"])
base_font_pt = st.sidebar.slider("Base font size (pt)", 18, 160, 42)

st.sidebar.header("Ink & appearance")
ink_color = st.sidebar.color_picker("Ink color", "#1b2a45")  # dark blue by default matching sample
paper_base_color = st.sidebar.color_picker("Paper base color", "#fbf8f3")
left_margin_color = st.sidebar.color_picker("Margin guideline color", "#d84f4f")
draw_margin = st.sidebar.checkbox("Draw left margin line (if not using paper background)", value=True)
notebook_lines = st.sidebar.checkbox("Draw ruled lines (if not using paper background)", value=True)

st.sidebar.header("Realism & variation (affects speed)")
jitter_px = st.sidebar.slider("Per-character jitter (px)", 0, 16, 4)
rotation_deg = st.sidebar.slider("Per-character rotation max (deg)", 0.0, 18.0, 6.0)
baseline_wobble_px = st.sidebar.slider("Baseline wobble (px)", 0, 14, 3)
ink_spread_px = st.sidebar.slider("Ink spread / blur (px)", 0, 6, 2)
smudge_strength = st.sidebar.slider("Smudge intensity (0-1)", 0.0, 1.0, 0.10, step=0.01)
pressure_variability = st.sidebar.slider("Pressure variability (0-1)", 0.0, 1.0, 0.35, step=0.01)

st.sidebar.header("Layout & paging")
margin_mm = st.sidebar.slider("Page margin (mm)", 5, 40, 16)
line_spacing_mult = st.sidebar.slider("Line spacing multiplier", 1.0, 2.0, 1.35, step=0.05)
indent_paragraph = st.sidebar.checkbox("Indent paragraphs", value=True)
max_pages = st.sidebar.slider("Max pages to generate", 1, 50, 8)

st.sidebar.header("Preview & generation")
preview_scale = st.sidebar.slider("Preview scale (0.2 = small)", 0.2, 1.0, 0.45)
generate_button = st.sidebar.button("✨ Generate Handwriting")

# ----------------------------
# Style presets
# ----------------------------
STYLE_PRESETS = {
    "Neat Cursive (slightly messy)": {
        "char_spacing_factor": 0.98,
        "rotation_factor": 0.6,
        "smear_probability": 0.04,
        "double_stroke_prob": 0.18,
        "random_word_shift": 0.06,
    },
    "Messy Cursive": {
        "char_spacing_factor": 1.05,
        "rotation_factor": 1.0,
        "smear_probability": 0.14,
        "double_stroke_prob": 0.35,
        "random_word_shift": 0.12,
    },
    "Natural Scribble": {
        "char_spacing_factor": 1.2,
        "rotation_factor": 1.5,
        "smear_probability": 0.28,
        "double_stroke_prob": 0.5,
        "random_word_shift": 0.18,
    },
}

preset = STYLE_PRESETS[style_choice]


# ----------------------------
# Text input (main)
# ----------------------------
st.markdown("### ✏️ Paste / type text to convert to handwriting")
text_input = st.text_area("Input text", height=360, placeholder="Start typing or paste long form text here...")
text_input = safe_truncate_text(text_input, max_chars=400000)  # guard memory


# ----------------------------
# Font loader (safe)
# ----------------------------
def load_font(uploaded, size_px: int) -> ImageFont.FreeTypeFont:
    """
    Load a font. If user uploaded one, store to a temp file and load.
    Else try several common fonts; fall back to PIL's default.
    size_px: pixel size (not pt)
    """
    if uploaded:
        ext = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            tmp_path = tmp.name
        try:
            f = ImageFont.truetype(tmp_path, size=size_px)
            return f
        except Exception:
            # fallback to default
            pass

    # Try common fonts that exist on many systems (Streamlit Cloud usually has DejaVu)
    candidates = ["DejaVuSerif.ttf", "DejaVuSans.ttf", "FreeSerif.ttf", "LiberationSerif-Regular.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size_px)
        except Exception:
            continue
    # final fallback
    return ImageFont.load_default()


# ----------------------------
# Notebook background drawing
# ----------------------------
def build_notebook_background(
    page_size_px: Tuple[int, int],
    dpi: int,
    paper_color_hex: str,
    left_margin_color_hex: str,
    draw_margin: bool,
    notebook_lines: bool,
    line_spacing_px: int,
    left_margin_px: int,
    uploaded_bg: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Returns an RGB PIL Image for the background.
    If uploaded_bg is provided, resize and use it. Otherwise draw clean paper with ruled lines
    and a margin.
    """
    W, H = page_size_px

    if uploaded_bg is not None:
        # Resize uploaded background to exactly page dimensions preserving aspect (crop/fit)
        bg = uploaded_bg.convert("RGB")
        bg = ImageOps.fit(bg, (W, H), method=Image.LANCZOS)
        return bg

    # Create base paper color
    base = Image.new("RGB", (W, H), paper_color_hex)
    draw = ImageDraw.Draw(base)

    # Draw small subtle paper texture (paper-color noise)
    # We'll add a lightweight per-pixel noise layer to avoid flat look
    arr = np.asarray(base).astype(np.int16)
    noise = (np.random.normal(loc=0.0, scale=3.0, size=arr.shape)).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    base = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(base)

    # Notebook lines
    if notebook_lines:
        # standard ruled spacing ~ 8 mm (but depends on design). We'll compute line spacing in px.
        # We get line_spacing_px as input based on the font's line height * multiplier.
        y = left_margin_px  # start after top margin equal to left margin (visually)
        line_color = tuple(int(left_margin_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        # Make the color slightly muted for ruling lines
        rule_rgb = (max(0, line_color[0] - 80), max(0, line_color[1] - 80), max(0, line_color[2] - 120))
        # line thickness scaled with DPI
        thickness = max(1, dpi // 300)
        for gy in range(left_margin_px, H - left_margin_px, line_spacing_px):
            draw.line((left_margin_px, gy, W - left_margin_px, gy), fill=(214, 208, 192), width=thickness)

    # Left margin/red line
    if draw_margin:
        mx = left_margin_px + int(0.5 * dpi / 72)  # small offset
        # darker red-ish
        try:
            red_rgb = tuple(int(left_margin_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            red_rgb = (200, 60, 60)
        thickness = max(1, int(2 * dpi / 300))
        draw.line((mx, left_margin_px, mx, H - left_margin_px), fill=red_rgb, width=thickness)

    return base


# ----------------------------
# Low-level glyph renderer
# ----------------------------
def render_character_glyph(
    ch: str,
    font: ImageFont.FreeTypeFont,
    ink_color_hex: str,
    pressure: float,
    stroke_thickness: float,
) -> Image.Image:
    """
    Render a single character to an RGBA image with simulated pressure (alpha) and thickness.
    We draw the glyph multiple times with tiny offsets to simulate thicker ink for higher pressure.
    """
    # Create a minimal image to draw the glyph into
    # Note: font.getsize is reliable; returns width,height
    try:
        gw, gh = font.getsize(ch)
    except Exception:
        # fallback
        gw, gh = font.getmask(ch).size

    # add padding for rotation and jitter
    pad = max(12, int(stroke_thickness * 3) + 6)
    img_w = gw + pad * 2
    img_h = gh + pad * 2
    rgba = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)

    # convert hex color + pressure to alpha
    base_r, base_g, base_b, _ = hex_to_rgba_tuple(ink_color_hex, 255)
    alpha = int(220 * (0.5 + 0.5 * pressure))  # pressure affects alpha

    # simulate thicker stroke by drawing glyph multiple times with small random offsets
    draws = max(1, int(round(stroke_thickness * (0.5 + pressure * 1.5))))
    for i in range(draws):
        ox = random.randint(-1, 1)
        oy = random.randint(-1, 1)
        draw.text((pad + ox, pad + oy), ch, font=font, fill=(base_r, base_g, base_b, alpha))

    return rgba


def apply_ink_spread_to_layer(layer: Image.Image, spread_px: float) -> Image.Image:
    """
    Simulate ink spread/bleed by blurring the alpha channel (and optionally darkening).
    """
    if spread_px <= 0.5:
        return layer
    # separate alpha, blur it, and recomposite
    rgba = layer.convert("RGBA")
    base_rgb = rgba.convert("RGB")
    alpha = rgba.split()[-1]
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=spread_px))
    # Reapply blurred alpha with original color multiply to simulate spread
    new = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    new.paste(base_rgb, (0, 0), blurred_alpha)
    return new


# ----------------------------
# High-level page renderer
# ----------------------------
def render_handwritten_pages(
    text: str,
    font: ImageFont.FreeTypeFont,
    paper_image: Image.Image,
    page_size_px: Tuple[int, int],
    margin_px: int,
    ink_color_hex: str,
    dpi: int,
    base_line_height_px: int,
    line_spacing_mult: float,
    preset: dict,
    jitter_px: int,
    rotation_deg: float,
    baseline_wobble_px: int,
    ink_spread_px: int,
    smudge_strength: float,
    pressure_variability: float,
    indent_paragraph: bool,
    max_pages: int,
) -> List[Image.Image]:
    """
    Create one or more PIL Image pages containing the hand-drawn-like text.
    Algorithm:
      - Break text into paragraphs and words
      - Place characters one-by-one with randomized position/rotation/pressure
      - If a character hits right margin, wrap to next line
      - If bottom margin exceeded, start a new page
      - After text is placed, post-process each page with ink spread, smudges, tone
    """

    W, H = page_size_px
    pages: List[Image.Image] = []
    text = text.replace("\r\n", "\n").rstrip("\n")
    paragraphs = text.split("\n\n")  # blank-line separated paragraphs

    # Clone background for page image
    def new_page():
        return paper_image.copy()

    cur_page = new_page()
    draw_page = ImageDraw.Draw(cur_page)

    # Compute usable area
    usable_w = W - 2 * margin_px
    usable_h = H - 2 * margin_px

    # Starting cursor (x,y)
    cursor_x = margin_px
    cursor_y = margin_px

    # Base metrics
    font_line_height = base_line_height_px
    # effective line height
    line_height = int(round(font_line_height * line_spacing_mult))

    # Optional top offset so lines align nicely with top ruled line
    # We'll snap to the first ruled line by adjusting cursor_y a bit when using drawn lines.
    # For uploaded background, we cannot detect lines — we keep margin-based start.

    # If indent enabled, use a tab-size indent in px
    indent_px = int(font_line_height * 1.0) if indent_paragraph else 0

    # Safeguard against infinite loops
    pages_generated = 0

    # Word wrapping helper — but we place char by char for realism
    def measure_word_px(word: str) -> int:
        try:
            return font.getsize(word)[0]
        except Exception:
            return font.getmask(word).size[0]

    # iterate paragraphs
    for para_idx, para in enumerate(paragraphs):
        if pages_generated >= max_pages:
            break
        para = para.strip("\n")
        # handle empty paragraph
        if para.strip() == "":
            cursor_y += int(line_height * 0.5)
            if cursor_y + line_height > margin_px + usable_h:
                # new page
                pages.append(cur_page)
                pages_generated += 1
                if pages_generated >= max_pages:
                    break
                cur_page = new_page()
                draw_page = ImageDraw.Draw(cur_page)
                cursor_y = margin_px
                cursor_x = margin_px
            continue

        words = para.split(" ")
        # Indent first line
        first_line = True
        for widx, word in enumerate(words):
            # The space after the word (unless last)
            space_after = " " if widx < len(words) - 1 else ""
            full_word = word + space_after

            # Pre-measure to see if it fits in remaining width — but we will place char-by-char;
            # if the pre-measure doesn't fit, break to next line.
            word_px = measure_word_px(full_word)
            if cursor_x + word_px > margin_px + usable_w:
                # Move to next line
                cursor_x = margin_px
                cursor_y += line_height
                first_line = False
                if cursor_y + line_height > margin_px + usable_h:
                    pages.append(cur_page)
                    pages_generated += 1
                    if pages_generated >= max_pages:
                        break
                    cur_page = new_page()
                    draw_page = ImageDraw.Draw(cur_page)
                    cursor_y = margin_px
                    cursor_x = margin_px
            if pages_generated >= max_pages:
                break

            # If it's the first line and indent is enabled and at start, add indent
            if first_line and indent_paragraph and cursor_x == margin_px:
                cursor_x += indent_px

            # Place each character in the full_word
            for ci, ch in enumerate(full_word):
                if pages_generated >= max_pages:
                    break

                # Variations: pressure, jitter, rotation
                pressure = max(0.1, min(1.0, random.random() * pressure_variability + (1.0 - pressure_variability)))
                stroke_thickness = 1.0 + pressure * 1.5  # base thickness
                glyph = render_character_glyph(ch, font, ink_color, pressure, stroke_thickness)

                # Optionally apply ink spread to the glyph layer
                glyph = apply_ink_spread_to_layer(glyph, spread_px=ink_spread_px * (0.5 + pressure))

                # Choose random jitter and rotation per glyph
                jx = int(round(random.uniform(-jitter_px, jitter_px) * (0.6 + random.random() * preset["char_spacing_factor"])))
                jy = int(round(random.uniform(-baseline_wobble_px, baseline_wobble_px)))
                rot = random.uniform(-rotation_deg, rotation_deg) * preset["rotation_factor"]

                # Random vertical "word shift" for messy look
                word_shift = int(round(random.uniform(-1, 1) * preset["random_word_shift"] * font_line_height))
                jy += word_shift

                # Rotate glyph
                glyph = glyph.rotate(rot, resample=Image.BICUBIC, expand=True)

                # Compute paste position (accounting for glyph padding)
                # The glyph image has internal padding; we paste so that its left aligns with cursor_x + jx
                # We also adjust y by -pad to keep it on baseline. We will use the glyph height to align baseline.
                gw, gh = glyph.size
                paste_x = cursor_x + jx
                paste_y = cursor_y + jy

                # If pasting would overflow right edge, wrap
                if paste_x + gw > margin_px + usable_w:
                    cursor_x = margin_px
                    cursor_y += line_height
                    paste_x = cursor_x + jx
                    paste_y = cursor_y + jy
                    first_line = False
                    if cursor_y + line_height > margin_px + usable_h:
                        pages.append(cur_page)
                        pages_generated += 1
                        if pages_generated >= max_pages:
                            break
                        cur_page = new_page()
                        draw_page = ImageDraw.Draw(cur_page)
                        cursor_y = margin_px
                        cursor_x = margin_px
                        paste_x = cursor_x + jx
                        paste_y = cursor_y + jy

                # Paste the glyph (RGBA paste uses glyph alpha)
                try:
                    cur_page.paste(glyph, (int(paste_x), int(paste_y)), glyph)
                except Exception:
                    # fallback: paste without mask
                    cur_page.paste(glyph, (int(paste_x), int(paste_y)))

                # Advance cursor_x by glyph width scaled by spacing factor and a small randomness
                advance = max(1, int((gw * 0.7) * (1.0 + random.uniform(-0.06, 0.06)) * preset["char_spacing_factor"]))
                cursor_x += advance

            # small extra space after word (already included as space char)
            # continue to next word

            first_line = False

        # After paragraph, move to next line
        cursor_x = margin_px
        cursor_y += line_height

        # new page if needed
        if cursor_y + line_height > margin_px + usable_h:
            pages.append(cur_page)
            pages_generated += 1
            if pages_generated >= max_pages:
                break
            cur_page = new_page()
            draw_page = ImageDraw.Draw(cur_page)
            cursor_x = margin_px
            cursor_y = margin_px

    # Append final page if any text has been drawn into it (or if no pages yet)
    if pages_generated < max_pages:
        pages.append(cur_page)
        pages_generated += 1

    # Limit to max_pages
    pages = pages[:max_pages]

    # Post-process pages: overall ink blur/spread and smudges
    processed_pages: List[Image.Image] = []
    for p in pages:
        # Convert to RGBA for composite effects
        p_rgba = p.convert("RGBA")

        # Lightly blur / blend to simulate paper absorption; controlled by ink_spread_px
        if ink_spread_px > 0:
            blurred = p_rgba.filter(ImageFilter.GaussianBlur(radius=ink_spread_px * 0.4))
            # blend the blurred with original to soften edges a bit
            p_rgba = Image.blend(p_rgba, blurred, alpha=0.18)

        # Add smudges: small translucent blobs of ink blurred
        if smudge_strength > 0:
            smudge_layer = Image.new("RGBA", p_rgba.size, (0, 0, 0, 0))
            sd = int(max(1, smudge_strength * 8))
            Wp, Hp = p_rgba.size
            # Number of smudges proportional to smudge_strength and page size
            n_smudges = int(1 + smudge_strength * 12)
            smd = ImageDraw.Draw(smudge_layer)
            for _ in range(n_smudges):
                rx = random.randint(0, Wp)
                ry = random.randint(0, Hp)
                r = int(max(6, min(Wp, Hp) * (0.01 + random.random() * 0.03 * smudge_strength)))
                alpha = int(40 * smudge_strength * random.random())
                # draw an elliptical smudge
                color = hex_to_rgba_tuple(ink_color, alpha)[0:3] + (alpha,)
                smd.ellipse([rx - r, ry - r, rx + r, ry + r], fill=color)
            smudge_layer = smudge_layer.filter(ImageFilter.GaussianBlur(radius=3 * smudge_strength + 0.5))
            p_rgba = Image.alpha_composite(p_rgba, smudge_layer)

        # Slightly add paper grain using a light noise overlay
        grain = Image.effect_noise(p_rgba.size, 6.0)
        grain = grain.convert("L").resize(p_rgba.size)
        grain = ImageOps.colorize(grain, black="#ffffff", white="#000000")
        grain = Image.blend(p_rgba.convert("RGB"), grain.convert("RGB"), alpha=0.02)
        p_final = Image.blend(p_rgba.convert("RGB"), grain, alpha=0.02)

        processed_pages.append(p_final)

    return processed_pages


# ----------------------------
# Multi-page PDF exporter
# ----------------------------
def pages_to_multi_pdf_bytes(pages: List[Image.Image]) -> bytes:
    """Return bytes of a multi-page PDF built from pages (PIL Images)."""
    if not pages:
        return b""
    rgb_pages = [p.convert("RGB") for p in pages]
    out = io.BytesIO()
    if len(rgb_pages) == 1:
        rgb_pages[0].save(out, format="PDF")
    else:
        rgb_pages[0].save(out, format="PDF", save_all=True, append_images=rgb_pages[1:])
    return out.getvalue()


# ----------------------------
# Main generation trigger
# ----------------------------
if generate_button:
    if not text_input.strip():
        st.error("Please enter some text to render.")
    else:
        with st.spinner("Rendering high-quality handwriting pages — this may take a few seconds..."):
            # Compute page size in pixels
            W_px = mm_to_px(PAPER_SIZES_MM[paper_choice][0], dpi)
            H_px = mm_to_px(PAPER_SIZES_MM[paper_choice][1], dpi)
            page_size_px = (W_px, H_px)

            # Convert base font pt to px for given DPI
            font_size_px = pt_to_px(base_font_pt, dpi)

            # Load font
            pil_font = load_font(font_upload, font_size_px)

            # Base line height — use font metrics where available
            try:
                # font.getsize returns (w,h)
                sample_w, sample_h = pil_font.getsize("Hg")
                base_line_height_px = max(sample_h, int(font_size_px * 1.05))
            except Exception:
                base_line_height_px = int(font_size_px * 1.25)

            # Left margin
            margin_px = mm_to_px(margin_mm, dpi)

            # Create or load paper background
            uploaded_bg_img = None
            if use_paper_bg and paper_bg_upload is not None:
                try:
                    uploaded_bg_img = Image.open(io.BytesIO(paper_bg_upload.read()))
                except Exception:
                    uploaded_bg_img = None

            paper_img = build_notebook_background(
                page_size_px=page_size_px,
                dpi=dpi,
                paper_color_hex=paper_base_color,
                left_margin_color_hex=left_margin_color,
                draw_margin=draw_margin,
                notebook_lines=notebook_lines,
                line_spacing_px=int(round(base_line_height_px * line_spacing_mult)),
                left_margin_px=margin_px,
                uploaded_bg=uploaded_bg_img,
            )

            # Finally render pages
            pages = render_handwritten_pages(
                text=text_input,
                font=pil_font,
                paper_image=paper_img,
                page_size_px=page_size_px,
                margin_px=margin_px,
                ink_color_hex=ink_color,
                dpi=dpi,
                base_line_height_px=base_line_height_px,
                line_spacing_mult=line_spacing_mult,
                preset=preset,
                jitter_px=jitter_px,
                rotation_deg=rotation_deg,
                baseline_wobble_px=baseline_wobble_px,
                ink_spread_px=ink_spread_px,
                smudge_strength=smudge_strength,
                pressure_variability=pressure_variability,
                indent_paragraph=indent_paragraph,
                max_pages=max_pages,
            )

            # Present preview and downloads
            st.success(f"Rendered {len(pages)} page(s).")

            # Layout: preview on left, controls on right
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("Preview (scaled)")
                for i, p in enumerate(pages):
                    # scaled preview
                    sw = int(p.size[0] * preview_scale)
                    sh = int(p.size[1] * preview_scale)
                    preview_img = p.resize((sw, sh), Image.LANCZOS)
                    st.image(preview_img, caption=f"Page {i+1}", use_column_width=False)

            with col_right:
                st.subheader("Download options")
                # Single page PNG (first)
                buf0 = io.BytesIO()
                pages[0].save(buf0, format="PNG")
                st.download_button("⬇️ Download Page 1 PNG", buf0.getvalue(), "handwritten_page_1.png", "image/png")

                # Per-page PNG downloads (if few pages)
                if len(pages) <= 12:
                    for i, p in enumerate(pages):
                        b = io.BytesIO()
                        p.save(b, format="PNG")
                        st.download_button(f"PNG: Page {i+1}", b.getvalue(), f"handwritten_page_{i+1}.png", "image/png")

                # Multi-page PDF
                pdf_bytes = pages_to_multi_pdf_bytes(pages)
                st.download_button("⬇️ Download multi-page PDF", pdf_bytes, "handwritten_pages.pdf", "application/pdf")

                # Offer a small info block with recommended settings for print
                st.markdown(
                    """
                    **Recommended for printing**
                    - DPI: 300 (default) or 600 for high detail.
                    - Preview scale: reduce so the browser stays responsive.
                    - If you plan to print, download the multi-page PDF.
                    """
                )

# ----------------------------
# Tips and Ethics reminder
# ----------------------------
st.markdown("---")
st.markdown(
    """
### Tips to improve realism
- Upload a handwriting-like TTF font for even better results. Combine that with `jitter`, `rotation`, and `ink spread`.
- Increase DPI to 600 for very high quality (note: generation will be slower).
- Slightly increase `smudge` and `ink spread` to mimic pen-bleed on porous paper.
- If you want a particular person's style, the correct approach is to practice/collect samples and train a model — do not impersonate.

**Ethics & legality**: generating handwriting is for creative and legitimate uses (mockups, art, practice). **Do not use** this tool to forge signatures, impersonate people, or misrepresent official documents.
"""
)

# End of file
