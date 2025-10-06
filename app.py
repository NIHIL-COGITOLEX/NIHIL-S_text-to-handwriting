# app.py
"""
Advanced Handwriting Generator (Streamlit)
------------------------------------------
- Uses a fixed realistic paper background image named `lined_paper.png` (placed in repo root)
  OR accepts an uploaded paper background via the sidebar.
- Produces A4 output at configurable DPI (default 300 DPI).
- Multi-page rendering with proper baseline/alignment using font.getmetrics().
- Realistic variability: per-character jitter, rotation, baseline wobble, pressure, ink spread, smudges.
- Exports PNG for each page and a single multi-page PDF for printing.
- Fixes the "half down / clipped letters" problem by aligning glyphs to computed baseline.
"""

import os
import io
import math
import random
import tempfile
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="✍️ Handwriting — A4 (Baseline-corrected)", layout="wide")
st.title("✍️ Handwriting Generator — A4 (Baseline-corrected)")
st.markdown(
    "Drop a realistic `lined_paper.png` (A4 at 300 DPI preferred) into the repo or upload a background image. "
    "**Do not use for forgery.**"
)

# ----------------------------
# Utilities: conversions & color
# ----------------------------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))

def pt_to_px(pt: float, dpi: int) -> int:
    return int(round(pt * dpi / 72.0))

def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch*2 for ch in h)
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (r, g, b, alpha)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Paper / Output")
paper_size = st.sidebar.selectbox("Paper size", ["A4"], index=0)  # we only support A4 here
dpi = st.sidebar.selectbox("DPI (print quality)", [150, 200, 300, 600], index=2)
st.sidebar.markdown("**Put `lined_paper.png` (A4 at desired DPI) in repo root for best results.**")
use_repo_bg = st.sidebar.checkbox("Use repo `lined_paper.png` if present", value=True)
uploaded_bg = st.sidebar.file_uploader("Upload paper background (optional)", type=["png","jpg","jpeg"])

st.sidebar.header("Handwriting style & font")
style = st.sidebar.selectbox("Style", ["Neat Cursive (slightly messy)", "Casual Messy Cursive", "Very Scribbled"])
font_upload = st.sidebar.file_uploader("Upload handwriting TTF/OTF (optional)", type=["ttf","otf"])
base_font_pt = st.sidebar.slider("Base font size (pt)", 20, 140, 48)

st.sidebar.header("Realism & alignment (fixes clipped letters)")
jitter_px = st.sidebar.slider("Per-character jitter (px)", 0, 20, 4)
rotation_deg = st.sidebar.slider("Rotation max (deg)", 0.0, 16.0, 6.0)
baseline_wobble = st.sidebar.slider("Baseline wobble (px)", 0, 18, 4)
ink_spread_px = st.sidebar.slider("Ink spread/blur (px)", 0, 6, 2)
smudge_strength = st.sidebar.slider("Smudge intensity (0-1)", 0.0, 1.0, 0.12)
pressure_variance = st.sidebar.slider("Pressure variance (0-1)", 0.0, 1.0, 0.35)

st.sidebar.header("Layout & pagination")
margin_mm = st.sidebar.slider("Page margin (mm)", 5, 40, 18)
line_spacing = st.sidebar.slider("Line spacing multiplier", 1.0, 2.0, 1.35, step=0.05)
indent_paragraph = st.sidebar.checkbox("Indent paragraphs", value=True)
max_pages = st.sidebar.slider("Max pages", 1, 50, 8)

st.sidebar.header("Preview / Export")
preview_scale = st.sidebar.slider("Preview scale (0.2 = small)", 0.2, 1.0, 0.45)
generate_button = st.sidebar.button("✨ Generate")

# ----------------------------
# Preset variables per style
# ----------------------------
STYLE_PRESETS = {
    "Neat Cursive (slightly messy)": dict(char_spacing_factor=0.96, rotation_factor=0.6, double_stroke_prob=0.18, smear_prob=0.04, vertical_jitter_factor=0.4),
    "Casual Messy Cursive": dict(char_spacing_factor=1.05, rotation_factor=1.0, double_stroke_prob=0.35, smear_prob=0.12, vertical_jitter_factor=0.8),
    "Very Scribbled": dict(char_spacing_factor=1.2, rotation_factor=1.4, double_stroke_prob=0.6, smear_prob=0.28, vertical_jitter_factor=1.25),
}
preset = STYLE_PRESETS[style]

# ----------------------------
# Text input
# ----------------------------
st.markdown("### Input text")
raw_text = st.text_area("Enter or paste the text you want written (blank lines = paragraph breaks)", height=300)
raw_text = raw_text.strip()

# ----------------------------
# Font loader & metrics helpers
# ----------------------------
def load_font(uploaded_file, size_px):
    """Load font from uploaded file or fallback to common fonts available in many environments."""
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            path = tmp.name
        try:
            return ImageFont.truetype(path, size_px)
        except Exception as e:
            st.warning(f"Uploaded font could not be loaded: {e}. Falling back.")
    # Try common fonts
    candidates = ["DejaVuSerif.ttf", "DejaVuSans.ttf", "LiberationSerif-Regular.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size_px)
        except Exception:
            continue
    return ImageFont.load_default()

def get_font_metrics(font: ImageFont.FreeTypeFont) -> Tuple[int,int]:
    """
    Return ascent and descent in pixels for a given PIL font.
    ascent: distance above baseline, descent: (positive) distance below baseline.
    """
    try:
        ascent, descent = font.getmetrics()
        return ascent, descent
    except Exception:
        # fallback heuristics
        try:
            bbox = font.getbbox("Hg")
            return bbox[3], abs(bbox[1])
        except Exception:
            return int(font.size * 0.8), int(font.size * 0.2)

# ----------------------------
# Paper background loader
# ----------------------------
def load_repo_background_if_exists(page_px: Tuple[int,int]) -> Optional[Image.Image]:
    # look for 'lined_paper.png' in repo root (app folder)
    if not use_repo_bg:
        return None
    candidates = ["lined_paper.png", "lined_paper.jpg", "lined_paper.jpeg"]
    for c in candidates:
        if os.path.exists(c):
            try:
                im = Image.open(c).convert("RGB")
                # Fit/crop to page size
                return ImageOps.fit(im, page_px, Image.LANCZOS)
            except Exception:
                continue
    return None

def load_background(page_px: Tuple[int,int]) -> Image.Image:
    # priority: uploaded_bg (sidebar) => repo file => generated blank paper with subtle texture
    if uploaded_bg is not None:
        try:
            uploaded_bg.seek(0)
            im = Image.open(uploaded_bg).convert("RGB")
            return ImageOps.fit(im, page_px, Image.LANCZOS)
        except Exception:
            st.warning("Uploaded paper background could not be loaded; fallback will be used.")
    repo_bg = load_repo_background_if_exists(page_px)
    if repo_bg is not None:
        return repo_bg
    # generate subtle textured paper + faint ruled lines (if no background provided)
    W, H = page_px
    base = Image.new("RGB", (W, H), "#FBF8F3")
    # subtle noise
    arr = np.asarray(base).astype(np.int16)
    noise = (np.random.normal(0.0, 3.5, arr.shape)).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    p = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(p)
    # draw faint lines in a pattern similar to notebook (8mm spacing approximated)
    approx_mm = 8.0
    spacing = mm_to_px(approx_mm, dpi)
    line_color = (210, 205, 192)
    for y in range(int(spacing/2), H, spacing):
        draw.line((int(margin_px/2), y, W - int(margin_px/2), y), fill=line_color, width=max(1, dpi//300))
    # left margin
    mx = int(margin_px * 0.75)
    draw.line((mx, int(margin_px/2), mx, H - int(margin_px/2)), fill=(220,110,110), width=max(1, dpi//280))
    return p

# ----------------------------
# Glyph painter (per character) with baseline alignment fix
# ----------------------------
def render_char_to_image(ch: str, font: ImageFont.FreeTypeFont, ink_hex: str, pressure: float, stroke_base: float) -> Image.Image:
    """
    Render a single character to an RGBA image.
    We place the glyph inside a padded box so after rotation/blur it won't clip.
    The glyph is drawn with pressure-dependent alpha & simulated thickness via multiple draws.
    """
    # metrics to choose box size
    try:
        gw, gh = font.getsize(ch)
    except Exception:
        gw, gh = font.getmask(ch).size

    pad = max(10, int(stroke_base * 3) + 8)
    W = gw + pad * 2
    H = gh + pad * 2
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(layer)

    r,g,b,_ = hex_to_rgba(ink_hex, 255)
    # alpha based on pressure; darker when pressure is high
    alpha = int(200 * (0.45 + 0.55 * pressure))
    # multiple draws to fake thicker stroke
    draws = max(1, int(round(stroke_base * (0.8 + pressure * 1.6))))
    for i in range(draws):
        # tiny offset for naturalness
        ox = random.randint(-1,1)
        oy = random.randint(-1,1)
        draw.text((pad + ox, pad + oy), ch, font=font, fill=(r,g,b,alpha))

    return layer

def apply_blur_alpha(layer: Image.Image, spread_px: float) -> Image.Image:
    """Blur alpha channel to simulate bleeding/ink spread."""
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
# High-level page renderer (multi-page)
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
    pressure_variance: float,
    indent_paragraph: bool,
    preset: dict,
    max_pages: int,
    paper_image: Image.Image,
) -> List[Image.Image]:
    """
    Render the entire text into one or more pages.
    Baseline fix strategy:
      - Use font.getmetrics() to compute ascent & descent.
      - The baseline for each line is set so that glyphs with ascenders / descenders do not clip.
      - When pasting a glyph layer (which contains padding), compute the top Y such that:
          paste_y = baseline_y - ascent - pad
    """
    W, H = page_px
    pages: List[Image.Image] = []

    if not text:
        return [paper_image.copy()]

    # split paragraphs; double newline indicates paragraph break
    paragraphs = [p for p in text.split("\n\n")]

    # get font metrics
    ascent, descent = get_font_metrics(font)
    # font internal recommended line height
    base_line_height = ascent + descent
    if base_line_height < 10:
        base_line_height = int(font.size * 1.25)

    # Effective line height with multiplier
    line_height = int(round(base_line_height * line_spacing_mult))

    usable_w = W - 2*margin_px
    usable_h = H - 2*margin_px

    # prepare first page
    current_page = paper_image.copy()
    cursor_x = margin_px
    cursor_y = margin_px  # this will represent the top of the line box, baseline will be computed from it
    pages_generated = 0

    def new_page():
        nonlocal current_page, cursor_x, cursor_y, pages_generated
        pages.append(current_page)
        pages_generated += 1
        current_page = paper_image.copy()
        cursor_x = margin_px
        cursor_y = margin_px

    # helpers to measure plain word width using font
    def measure_text_px(t: str) -> int:
        try:
            return font.getsize(t)[0]
        except Exception:
            return font.getmask(t).size[0]

    # place function for single glyph with baseline alignment
    def paste_glyph(page: Image.Image, glyph_img: Image.Image, baseline_y: int, pad: int, paste_x: int, paste_extra_y: int = 0):
        # glyph_img was rendered with padding pad; to align baseline:
        # glyph_top_y = baseline_y - ascent - pad + paste_extra_y
        top_y = int(round(baseline_y - ascent - pad + paste_extra_y))
        # If glyph would go out of bottom, we signal for a wrap / new page externally
        page.paste(glyph_img, (int(paste_x), int(top_y)), glyph_img)

    # iterate paragraphs and words
    for pidx, para in enumerate(paragraphs):
        if pages_generated >= max_pages:
            break
        para = para.replace("\r\n", "\n").strip()
        if para == "":
            # blank paragraph -> vertical gap
            cursor_y += int(line_height * 0.6)
            if cursor_y + line_height > margin_px + usable_h:
                new_page()
            continue

        words = para.split(" ")
        first_word_of_line = True
        # apply paragraph indent
        para_indent_px = int(base_line_height * 1.0) if indent_paragraph else 0

        for widx, word in enumerate(words):
            if pages_generated >= max_pages:
                break
            spacer = " " if widx < len(words)-1 else ""
            full_word = word + spacer
            word_px = measure_text_px(full_word)

            # If word doesn't fit on current line, wrap
            if cursor_x + word_px > margin_px + usable_w and not first_word_of_line:
                # new line
                cursor_x = margin_px
                cursor_y += line_height
                first_word_of_line = True
                if cursor_y + line_height > margin_px + usable_h:
                    new_page()
                    if pages_generated >= max_pages:
                        break

            # for first line of paragraph, if at line start, add indent
            if first_word_of_line and para_indent_px > 0 and cursor_x == margin_px:
                cursor_x += para_indent_px

            # Place word character-by-character to allow jitter and rotation
            for ci, ch in enumerate(full_word):
                if pages_generated >= max_pages:
                    break
                # Pressure simulation
                pressure = max(0.08, min(1.0, random.random() * pressure_variance + (1.0 - pressure_variance)))
                stroke_base = 1.0 + pressure * 1.2

                # Render glyph to image (with padding)
                glyph = render_char_to_image(ch, font, ink_color, pressure, stroke_base)
                # apply ink spread on glyph-level
                glyph = apply_blur_alpha(glyph, spread_px=ink_spread_px * (0.5 + pressure*0.6))

                # random jitter/rotation / vertical shift
                jx = int(round(random.uniform(-jitter_px, jitter_px) * (0.6 + random.random()*preset["char_spacing_factor"])))
                jy = int(round(random.uniform(-baseline_wobble, baseline_wobble) * (preset["vertical_jitter_factor"])))
                rot = random.uniform(-rotation_deg, rotation_deg) * preset["rotation_factor"]

                # rotate glyph (expand=True so it doesn't crop)
                glyph = glyph.rotate(rot, resample=Image.BICUBIC, expand=True)

                # After rotation, glyph has different pad; approximate pad used when rendering original
                # We used pad inside render_char_to_image. Let's inspect glyph size to compute paste.
                pad = max(10, int(stroke_base * 3) + 8)

                # compute baseline Y (we align baseline to the same baseline for that line)
                baseline_y = cursor_y + ascent  # cursor_y is the top of the line box; baseline = top + ascent

                # if placing this glyph would overflow right edge, wrap before placing
                if cursor_x + glyph.size[0] > margin_px + usable_w:
                    # wrap
                    cursor_x = margin_px
                    cursor_y += line_height
                    baseline_y = cursor_y + ascent
                    if cursor_y + line_height > margin_px + usable_h:
                        new_page()
                        if pages_generated >= max_pages:
                            break

                # paste glyph with extra jitter added to vertical position
                paste_extra_y = jy
                paste_x = cursor_x + jx

                paste_glyph(current_page, glyph, baseline_y, pad, paste_x, paste_extra_y)

                # advance cursor_x — base on glyph width scaled with spacing factor and a bit randomness
                advance = max(1, int((glyph.size[0] * 0.75) * (1.0 + random.uniform(-0.06, 0.06)) * preset["char_spacing_factor"]))
                cursor_x += advance

            first_word_of_line = False

        # After paragraph, line break
        cursor_x = margin_px
        cursor_y += line_height
        if cursor_y + line_height > margin_px + usable_h:
            new_page()

    # make sure to append the last page
    if pages_generated < max_pages:
        pages.append(current_page)
        pages_generated += 1

    # Post-process pages: ink blur, smudges, paper grain (subtle)
    processed_pages: List[Image.Image] = []
    for p in pages[:max_pages]:
        p_rgba = p.convert("RGBA")
        # global blur/soften for ink spread
        if ink_spread_px > 0:
            blurred = p_rgba.filter(ImageFilter.GaussianBlur(radius=ink_spread_px * 0.4))
            p_rgba = Image.blend(p_rgba, blurred, alpha=0.10)

        # add smudge layer
        if smudge_strength > 0:
            sm_layer = Image.new("RGBA", p_rgba.size, (0,0,0,0))
            sd = ImageDraw.Draw(sm_layer)
            Wp, Hp = p_rgba.size
            n_smudges = int(1 + smudge_strength * 12)
            for _ in range(n_smudges):
                rx = random.randint(0, Wp)
                ry = random.randint(0, Hp)
                r = int(max(6, min(Wp, Hp) * (0.01 + random.random() * 0.03 * smudge_strength)))
                alpha = int(28 * smudge_strength * random.random())
                c = hex_to_rgba(ink_color, alpha)
                sd.ellipse([rx-r, ry-r, rx+r, ry+r], fill=c)
            sm_layer = sm_layer.filter(ImageFilter.GaussianBlur(radius=3 * smudge_strength + 0.5))
            p_rgba = Image.alpha_composite(p_rgba, sm_layer)

        # subtle paper grain
        arr = np.asarray(p_rgba.convert("RGB")).astype(np.float32)
        grain = (np.random.normal(0.0, 1.2, arr.shape)).astype(np.float32)
        arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
        p_final = Image.fromarray(arr, "RGB")
        processed_pages.append(p_final)

    return processed_pages

# ----------------------------
# PDF helper
# ----------------------------
def pages_to_pdf_bytes(pages: List[Image.Image]) -> bytes:
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
# Main generate action
# ----------------------------
if generate_button:
    if not raw_text:
        st.error("Please enter the text to render.")
    else:
        # A4 mm dims
        A4_mm = (210.0, 297.0)
        W_px = mm_to_px(A4_mm[0], dpi)
        H_px = mm_to_px(A4_mm[1], dpi)
        page_px = (W_px, H_px)

        # margin in px
        margin_px = mm_to_px(margin_mm, dpi)

        # compute font pixel size from pt
        font_px = pt_to_px(base_font_pt, dpi)
        font = load_font(font_upload, font_px)

        # compute ascent/descent, line_height
        ascent, descent = get_font_metrics(font)
        base_line_height = ascent + descent
        if base_line_height < 8:
            base_line_height = int(font_px * 1.25)

        # prepare paper background
        # first try repo file if user asked for that
        paper_img = load_background(page_px)

        # finally render pages
        with st.spinner("Rendering pages — this can take a few seconds depending on DPI & length..."):
            pages = render_pages(
                text=raw_text,
                font=font,
                page_px=page_px,
                margin_px=margin_px,
                line_spacing_mult=line_spacing,
                ink_hex="#1b2a45",  # default dark blue
                dpi=dpi,
                jitter_px=jitter_px,
                rotation_deg=rotation_deg,
                baseline_wobble=baseline_wobble,
                ink_spread_px=ink_spread_px,
                smudge_strength=smudge_strength,
                pressure_variance=pressure_variance,
                indent_paragraph=indent_paragraph,
                preset=preset,
                max_pages=max_pages,
                paper_image=paper_img,
            )

        st.success(f"Rendered {len(pages)} page(s).")

        # Show preview & downloads
        left_col, right_col = st.columns([2,1])
        with left_col:
            st.subheader("Preview (scaled)")
            for i, p in enumerate(pages):
                sw = int(p.size[0] * preview_scale)
                sh = int(p.size[1] * preview_scale)
                preview = p.resize((sw, sh), Image.LANCZOS)
                st.image(preview, caption=f"Page {i+1}", use_column_width=False)

        with right_col:
            st.subheader("Downloads")
            # Single PNG first page
            buf0 = io.BytesIO()
            pages[0].save(buf0, format="PNG")
            st.download_button("⬇️ Download Page 1 PNG", buf0.getvalue(), "handwritten_page_1.png", "image/png")
            # Per-page PNGs (if not too many)
            if len(pages) <= 12:
                for i, p in enumerate(pages):
                    b = io.BytesIO()
                    p.save(b, format="PNG")
                    st.download_button(f"PNG: Page {i+1}", b.getvalue(), f"handwritten_page_{i+1}.png", "image/png")
            # Multi-page PDF
            pdf_bytes = pages_to_pdf_bytes(pages)
            st.download_button("⬇️ Download multi-page PDF", pdf_bytes, "handwritten_pages.pdf", "application/pdf")
            st.markdown("**Tips:** Use 300 DPI for print. For higher fidelity use 600 DPI but it will be slower and use more memory.")

# ----------------------------
# Footer: help & tips
# ----------------------------
st.markdown("---")
st.markdown(
    """
**Why this avoids the 'half down / clipped' problem**
- We compute `ascent` and `descent` via `font.getmetrics()` and set the line baseline as `cursor_y + ascent`.
- Each glyph image is created with internal padding and pasted with `paste_y = baseline - ascent - pad`.
  This ensures ascenders & descenders have room and won't be clipped.
  
**Best practice**
- Provide a high-quality `lined_paper.png` in the repo root that is at least A4@300DPI (2480×3508 px).
  Name it exactly `lined_paper.png`.
- Upload a handwriting `.ttf` for best, most realistic results.
- Tweak `jitter`, `rotation`, `baseline wobble`, `ink spread` & `smudge` sliders to achieve the desired realism.

**Ethics:** This tool is for creative uses only. Do not use it for forgery, impersonation, or illegal documents.
"""
)

# End of file
