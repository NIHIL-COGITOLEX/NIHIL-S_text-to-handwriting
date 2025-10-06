# app.py
"""
Advanced Text → Handwriting (Streamlit)
- Uses a fixed paper background image if available (assets/backgrounds/lined_paper.png).
- Detects fonts in assets/fonts/ or accepts uploaded fonts.
- Baseline-correct rendering using font metrics (avoids clipped ascenders/descenders).
- Realism: jitter, rotation, baseline wobble, pressure/ink thickness, ink spread & smudges.
- Multi-page A4 output (PNG per page + multi-page PDF). ZIP option for all PNGs.
"""

import os
import io
import math
import random
import tempfile
import zipfile
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="✍️ Advanced Handwriting (A4)", layout="wide")
st.title("✍️ Advanced Handwriting Generator — A4 (Baseline-corrected)")
st.markdown(
    "Drop a high-res `lined_paper.png` in `assets/backgrounds/` or upload one. "
    "**For creative use only — do not forge or impersonate.**"
)

# ----------------------------
# Paths & auto-detect assets
# ----------------------------
ASSETS_FONTS_DIR = os.path.join("assets", "fonts")
ASSETS_BG_DIR = os.path.join("assets", "backgrounds")
REPO_BG_FILENAME = os.path.join(ASSETS_BG_DIR, "lined_paper.png")

os.makedirs(ASSETS_FONTS_DIR, exist_ok=True)
os.makedirs(ASSETS_BG_DIR, exist_ok=True)

# ----------------------------
# Utility helpers
# ----------------------------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))

def pt_to_px(pt: float, dpi: int) -> int:
    return int(round(pt * dpi / 72.0))

def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r = int(h[0:2], 16); g = int(h[2:4],16); b = int(h[4:6],16)
    return (r, g, b, alpha)

def list_fonts_in_assets() -> List[str]:
    """Return absolute paths to TTF/OTF font files inside assets/fonts/"""
    out = []
    for root, _, files in os.walk(ASSETS_FONTS_DIR):
        for f in files:
            if f.lower().endswith((".ttf", ".otf")):
                out.append(os.path.join(root, f))
    return sorted(out)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Paper / Output")
dpi = st.sidebar.selectbox("DPI (print quality)", [150, 200, 300, 600], index=2)
st.sidebar.markdown("Place `lined_paper.png` in `assets/backgrounds/` (recommended) or upload below.")
use_repo_bg = st.sidebar.checkbox("Prefer repo `assets/backgrounds/lined_paper.png` if present", value=True)
upload_bg = st.sidebar.file_uploader("Upload paper background (optional)", type=["png","jpg","jpeg"])

st.sidebar.header("Handwriting & font")
# fonts detected in assets
detected_fonts = list_fonts_in_assets()
font_choice_label = ["(upload or choose detected)"]
font_choice_label += [os.path.basename(p) for p in detected_fonts]
selected_font_name = st.sidebar.selectbox("Choose font (or upload)", font_choice_label)
uploaded_font = st.sidebar.file_uploader("Or upload a handwriting font (.ttf/.otf)", type=["ttf","otf"])
base_font_pt = st.sidebar.slider("Base font size (pt)", 18, 160, 48)

st.sidebar.header("Style preset")
style = st.sidebar.selectbox("Style preset", ["Neat Cursive", "Slightly Messy Cursive", "Fast Messy Scribble"])

st.sidebar.header("Ink & Paper look")
ink_color = st.sidebar.color_picker("Ink color", "#1b2a45")
paper_tint = st.sidebar.color_picker("Paper tint (subtle)", "#fbf8f3")
draw_margin_line = st.sidebar.checkbox("Ensure left margin line drawn (if background lacks one)", value=False)

st.sidebar.header("Realism / Tunables")
jitter_px = st.sidebar.slider("Per-character jitter (px)", 0, 20, 4)
rotation_deg = st.sidebar.slider("Per-character rotation (deg)", 0.0, 18.0, 5.0)
baseline_wobble = st.sidebar.slider("Baseline wobble (px)", 0, 18, 3)
ink_spread_px = st.sidebar.slider("Ink spread / bleed (px)", 0, 6, 2)
smudge_strength = st.sidebar.slider("Smudge intensity (0-1)", 0.0, 1.0, 0.12)
pressure_variability = st.sidebar.slider("Pressure variability (0-1)", 0.0, 1.0, 0.35)
stroke_thickness_variation = st.sidebar.slider("Stroke thickness variation (0-1)", 0.0, 1.0, 0.35)

st.sidebar.header("Layout & pagination")
margin_mm = st.sidebar.slider("Page margin (mm)", 6, 40, 18)
line_spacing_mult = st.sidebar.slider("Line spacing multiplier", 1.0, 2.0, 1.35, step=0.05)
indent_paragraph = st.sidebar.checkbox("Indent paragraphs", value=True)
max_pages = st.sidebar.slider("Max pages to generate", 1, 30, 6)

st.sidebar.header("Preview / Export")
preview_scale = st.sidebar.slider("Preview scale", 0.2, 1.0, 0.45)
generate_btn = st.sidebar.button("✨ Generate Handwriting")

# ----------------------------
# Style presets internal mapping
# ----------------------------
STYLE_MAP = {
    "Neat Cursive": dict(char_spacing_factor=0.96, rotation_factor=0.6, smear_prob=0.04, double_stroke_prob=0.18, vertical_shift=0.4),
    "Slightly Messy Cursive": dict(char_spacing_factor=1.05, rotation_factor=1.0, smear_prob=0.12, double_stroke_prob=0.30, vertical_shift=0.8),
    "Fast Messy Scribble": dict(char_spacing_factor=1.20, rotation_factor=1.5, smear_prob=0.28, double_stroke_prob=0.55, vertical_shift=1.2),
}
preset = STYLE_MAP[style]

# ----------------------------
# Main text input
# ----------------------------
st.markdown("### ✏️ Paste / type your text")
text_input = st.text_area("Input text (double newlines for paragraph breaks)", height=360, placeholder="Type or paste here...")
text_input = text_input.strip()

# ----------------------------
# Font loading & metrics helpers
# ----------------------------
def load_font_from_uploaded(uploaded_file, size_px: int):
    if not uploaded_file:
        return None
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_path = tmp.name
    try:
        return ImageFont.truetype(tmp_path, size_px)
    except Exception:
        return None

def load_chosen_font(size_px: int):
    # precedence: uploaded font -> selected repo font -> fallback common fonts -> PIL default
    if uploaded_font:
        f = load_font_from_uploaded(uploaded_font, size_px)
        if f:
            return f
    # selected detected font name could be the label string; check
    if selected_font_name and selected_font_name != "(upload or choose detected)":
        # find match from detected_fonts by basename
        for p in detected_fonts:
            if os.path.basename(p) == selected_font_name:
                try:
                    return ImageFont.truetype(p, size_px)
                except Exception:
                    break
    # try a few common fonts available on many environments
    for candidate in ["DejaVuSerif.ttf", "DejaVuSans.ttf", "FreeSerif.ttf", "LiberationSerif-Regular.ttf"]:
        try:
            return ImageFont.truetype(candidate, size_px)
        except Exception:
            continue
    return ImageFont.load_default()

def get_font_ascent_descent(font: ImageFont.FreeTypeFont) -> Tuple[int,int]:
    try:
        ascent, descent = font.getmetrics()
        return int(ascent), int(descent)
    except Exception:
        try:
            bbox = font.getbbox("Hg")
            return abs(bbox[3]), abs(bbox[1])
        except Exception:
            # fallback approx
            return int(font.size * 0.8), int(font.size * 0.25)

# ----------------------------
# Background loader
# ----------------------------
def load_background(page_px: Tuple[int,int]) -> Image.Image:
    W,H = page_px
    # priority: uploaded_bg -> repo file -> generated paper
    if upload_bg:
        try:
            upload_bg_bytes = upload_bg.read()
            im = Image.open(io.BytesIO(upload_bg_bytes)).convert("RGB")
            return ImageOps.fit(im, page_px, method=Image.LANCZOS)
        except Exception:
            st.warning("Uploaded background failed to load; falling back to repo/generated paper.")
    if use_repo_bg and os.path.exists(REPO_BG_FILENAME):
        try:
            im = Image.open(REPO_BG_FILENAME).convert("RGB")
            return ImageOps.fit(im, page_px, method=Image.LANCZOS)
        except Exception:
            st.warning("Repo background exists but could not be loaded.")
    # generate subtle paper with faint rules (fallback)
    base = Image.new("RGB", (W,H), paper_tint)
    arr = np.asarray(base).astype(np.int16)
    noise = (np.random.normal(0.0, 3.0, arr.shape)).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    p = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(p)
    # light ruled lines approx every 8mm
    spacing = mm_to_px(8.0, dpi)
    for y in range(spacing//2, H, spacing):
        draw.line((int(margin_px/2), y, W - int(margin_px/2), y), fill=(220,216,209), width=max(1, dpi//300))
    # margin line
    if draw_margin_line:
        mx = int(margin_px * 0.75)
        draw.line((mx, int(margin_px/2), mx, H - int(margin_px/2)), fill=(210,110,110), width=max(1, dpi//280))
    return p

# ----------------------------
# Low-level glyph renderers
# ----------------------------
def render_char_layer(ch: str, font: ImageFont.FreeTypeFont, ink_hex: str, pressure: float, stroke_base=1.0) -> Image.Image:
    """
    Render the glyph into a padded RGBA image; multiple tiny draws to simulate thickness.
    """
    try:
        gw, gh = font.getsize(ch)
    except Exception:
        gw, gh = font.getmask(ch).size
    pad = max(12, int(stroke_base * 3) + 8)
    W = gw + pad * 2
    H = gh + pad * 2
    layer = Image.new("RGBA", (W,H), (0,0,0,0))
    draw = ImageDraw.Draw(layer)
    r,g,b,_ = hex_to_rgba(ink_hex, 255)
    alpha = int(220 * (0.45 + 0.55 * pressure))
    # number of micro-draws depends on stroke_base & pressure
    draws = max(1, int(round(stroke_base * (0.7 + pressure * 1.6))))
    for _ in range(draws):
        ox = random.randint(-1,1)
        oy = random.randint(-1,1)
        draw.text((pad + ox, pad + oy), ch, font=font, fill=(r,g,b,alpha))
    return layer

def blur_alpha(layer: Image.Image, spread_px: float) -> Image.Image:
    """Blur alpha to simulate ink soaking/spread."""
    if spread_px <= 0.5:
        return layer
    rgba = layer.convert("RGBA")
    rgb = rgba.convert("RGB")
    alpha = rgba.split()[-1]
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=spread_px))
    out = Image.new("RGBA", layer.size, (0,0,0,0))
    out.paste(rgb, (0,0), blurred_alpha)
    return out

# ----------------------------
# Page composition (high-level)
# ----------------------------
def render_pages(
    text: str,
    font: ImageFont.FreeTypeFont,
    page_px: Tuple[int,int],
    margin_px: int,
    line_spacing_mult: float,
    ink_hex: str,
    dpi: int,
    jitter_px: int,
    rotation_deg: float,
    baseline_wobble: int,
    ink_spread_px: int,
    smudge_strength: float,
    pressure_variability: float,
    stroke_variation: float,
    indent_paragraph: bool,
    preset: dict,
    max_pages: int,
    paper_img: Image.Image,
) -> List[Image.Image]:
    """
    Render the text into multiple pages. Use font metrics (ascent/descent) to compute baseline.
    """
    W,H = page_px
    pages: List[Image.Image] = []

    if not text:
        return [paper_img.copy()]

    paragraphs = [p for p in text.replace("\r\n", "\n").split("\n\n")]

    ascent, descent = get_font_ascent_descent(font)
    base_line_height = ascent + descent
    if base_line_height < 8:
        base_line_height = int(font.size * 1.25)
    line_height = int(round(base_line_height * line_spacing_mult))
    usable_w = W - 2*margin_px
    usable_h = H - 2*margin_px

    current = paper_img.copy()
    cursor_x = margin_px
    cursor_y = margin_px
    pages_count = 0

    def new_page():
        nonlocal current, cursor_x, cursor_y, pages_count
        pages.append(current)
        pages_count += 1
        current = paper_img.copy()
        cursor_x = margin_px
        cursor_y = margin_px

    def measure_text_px(t: str) -> int:
        try:
            return font.getsize(t)[0]
        except Exception:
            return font.getmask(t).size[0]

    for pidx, para in enumerate(paragraphs):
        if pages_count >= max_pages:
            break
        para = para.strip()
        if para == "":
            cursor_y += int(line_height * 0.6)
            if cursor_y + line_height > margin_px + usable_h:
                new_page()
            continue

        words = para.split(" ")
        first_word = True
        para_indent_px = int(base_line_height * 1.0) if indent_paragraph else 0

        for widx, word in enumerate(words):
            if pages_count >= max_pages:
                break
            spacer = " " if widx < len(words)-1 else ""
            full_word = word + spacer
            word_px = measure_text_px(full_word)

            # wrap if needed
            if cursor_x + word_px > margin_px + usable_w and not first_word:
                cursor_x = margin_px
                cursor_y += line_height
                first_word = True
                if cursor_y + line_height > margin_px + usable_h:
                    new_page()
                    if pages_count >= max_pages:
                        break

            if first_word and para_indent_px > 0 and cursor_x == margin_px:
                cursor_x += para_indent_px

            for ci, ch in enumerate(full_word):
                if pages_count >= max_pages:
                    break
                # pressure & stroke thickness
                pressure = max(0.08, min(1.0, random.random() * pressure_variability + (1.0 - pressure_variability)))
                stroke_base = 1.0 + stroke_variation * random.random() * 1.5

                glyph = render_char_layer(ch, font, ink_hex, pressure, stroke_base)
                glyph = blur_alpha(glyph, spread_px=ink_spread_px * (0.4 + pressure*0.6))

                # jitter/rotation/wobble
                jx = int(round(random.uniform(-jitter_px, jitter_px) * (0.6 + random.random()*preset["char_spacing_factor"])))
                jy = int(round(random.uniform(-baseline_wobble, baseline_wobble) * (preset["vertical_shift"])))
                rot = random.uniform(-rotation_deg, rotation_deg) * preset["rotation_factor"]

                # rotate glyph
                glyph = glyph.rotate(rot, resample=Image.BICUBIC, expand=True)

                # approximate padding used in render_char_layer
                pad = max(12, int(stroke_base * 3) + 8)
                # baseline_y
                baseline_y = cursor_y + ascent

                # wrap check
                if cursor_x + glyph.size[0] > margin_px + usable_w:
                    cursor_x = margin_px
                    cursor_y += line_height
                    baseline_y = cursor_y + ascent
                    if cursor_y + line_height > margin_px + usable_h:
                        new_page()
                        if pages_count >= max_pages:
                            break

                # paste
                paste_x = int(cursor_x + jx)
                paste_y = int(round(baseline_y - ascent - pad + jy))
                try:
                    current.paste(glyph, (paste_x, paste_y), glyph)
                except Exception:
                    current.paste(glyph, (paste_x, paste_y))

                # advance
                advance = max(1, int((glyph.size[0] * 0.75) * (1.0 + random.uniform(-0.06, 0.06)) * preset["char_spacing_factor"]))
                cursor_x += advance

            first_word = False

        # end paragraph
        cursor_x = margin_px
        cursor_y += line_height
        if cursor_y + line_height > margin_px + usable_h:
            new_page()

    # append last page
    if pages_count < max_pages:
        pages.append(current)
        pages_count += 1

    # postprocess: global ink spread + smudges + grain
    final_pages: List[Image.Image] = []
    for p in pages[:max_pages]:
        p_rgba = p.convert("RGBA")
        if ink_spread_px > 0:
            blurred = p_rgba.filter(ImageFilter.GaussianBlur(radius=ink_spread_px * 0.35))
            p_rgba = Image.blend(p_rgba, blurred, alpha=0.12)

        # smudges
        if smudge_strength > 0:
            sm = Image.new("RGBA", p_rgba.size, (0,0,0,0))
            sdraw = ImageDraw.Draw(sm)
            Wp, Hp = p_rgba.size
            n_sm = int(1 + smudge_strength * 14)
            for _ in range(n_sm):
                rx = random.randint(0, Wp)
                ry = random.randint(0, Hp)
                r = int(max(6, min(Wp, Hp) * (0.01 + random.random() * 0.03 * smudge_strength)))
                alpha = int(30 * smudge_strength * random.random())
                c = hex_to_rgba(ink_color, alpha)
                sdraw.ellipse([rx-r, ry-r, rx+r, ry+r], fill=c)
            sm = sm.filter(ImageFilter.GaussianBlur(radius=3 * smudge_strength + 0.6))
            p_rgba = Image.alpha_composite(p_rgba, sm)

        # grain
        arr = np.asarray(p_rgba.convert("RGB")).astype(np.float32)
        grain = (np.random.normal(0.0, 1.0, arr.shape)).astype(np.float32)
        arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
        p_final = Image.fromarray(arr, "RGB")
        final_pages.append(p_final)

    return final_pages

# ----------------------------
# Exports: PNG, PDF, ZIP
# ----------------------------
def pages_to_pdf_bytes(pages: List[Image.Image]) -> bytes:
    if not pages:
        return b""
    rgb = [p.convert("RGB") for p in pages]
    out = io.BytesIO()
    if len(rgb) == 1:
        rgb[0].save(out, format="PDF")
    else:
        rgb[0].save(out, format="PDF", save_all=True, append_images=rgb[1:])
    return out.getvalue()

def pages_to_zip_bytes(pages: List[Image.Image]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(pages):
            b = io.BytesIO()
            p.save(b, format="PNG")
            z.writestr(f"handwritten_page_{i+1}.png", b.getvalue())
    return out.getvalue()

# ----------------------------
# Main: generate when button pressed
# ----------------------------
if generate_btn:
    if not text_input:
        st.error("Please enter text to render.")
    else:
        with st.spinner("Rendering pages... this may take a few seconds depending on DPI & length"):
            # A4 dims in mm -> px
            A4_mm = (210.0, 297.0)
            W_px = mm_to_px(A4_mm[0], dpi)
            H_px = mm_to_px(A4_mm[1], dpi)
            page_px = (W_px, H_px)
            margin_px = mm_to_px(margin_mm, dpi)

            # font in px
            font_px = pt_to_px(base_font_pt, dpi)
            font = load_chosen_font(font_px)
            ascent, descent = get_font_ascent_descent(font)
            base_line_height = ascent + descent
            if base_line_height < 8:
                base_line_height = int(font_px * 1.25)

            # background
            paper_img = load_background(page_px)

            # render
            pages = render_pages(
                text=text_input,
                font=font,
                page_px=page_px,
                margin_px=margin_px,
                line_spacing_mult=line_spacing_mult,
                ink_hex=ink_color,
                dpi=dpi,
                jitter_px=jitter_px,
                rotation_deg=rotation_deg,
                baseline_wobble=baseline_wobble,
                ink_spread_px=ink_spread_px,
                smudge_strength=smudge_strength,
                pressure_variability=pressure_variability,
                stroke_variation=stroke_thickness_variation,
                indent_paragraph=indent_paragraph,
                preset=preset,
                max_pages=max_pages,
                paper_img=paper_img,
            )

        st.success(f"Rendered {len(pages)} page(s).")

        # Preview and downloads
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Preview")
            for idx, p in enumerate(pages):
                sw = int(p.size[0] * preview_scale)
                sh = int(p.size[1] * preview_scale)
                preview = p.resize((sw, sh), Image.LANCZOS)
                st.image(preview, caption=f"Page {idx+1}", use_column_width=False)

        with col2:
            st.subheader("Download")
            # Page 1 PNG
            buf0 = io.BytesIO()
            pages[0].save(buf0, format="PNG")
            st.download_button("⬇️ Download Page 1 PNG", buf0.getvalue(), "handwritten_page_1.png", "image/png")

            # Multi-page PDF
            pdf_bytes = pages_to_pdf_bytes(pages)
            st.download_button("⬇️ Download multi-page PDF", pdf_bytes, "handwritten_pages.pdf", "application/pdf")

            # If few pages, provide individual PNGs and ZIP
            if len(pages) <= 12:
                for i,p in enumerate(pages):
                    bb = io.BytesIO()
                    p.save(bb, format="PNG")
                    st.download_button(f"PNG: Page {i+1}", bb.getvalue(), f"handwritten_page_{i+1}.png", "image/png")
            else:
                zip_bytes = pages_to_zip_bytes(pages)
                st.download_button("⬇️ Download all pages (ZIP)", zip_bytes, "handwritten_pages_png.zip", "application/zip")

        st.markdown("---")
        st.markdown(
            """
            **Notes & tips**
            - Put high-res `lined_paper.png` (A4 @ 300 DPI --> 2480×3508 px) in `assets/backgrounds/` for best authenticity.
            - Upload a handwriting `.ttf` (Dancing Script, Patrick Hand, Homemade Apple, etc.) to assets/fonts or upload in the sidebar.
            - Increase DPI (300 or 600) for print. Be mindful of memory/time.
            - If ascenders/descenders still clip for a specific font, slightly increase page margin or line spacing.
            """
        )
