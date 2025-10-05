import streamlit as st
import os
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from colorizers.mvgd import TransferMVGD  # ensure mvgd.py is in your project folder
from io import BytesIO
from PIL import Image
# -------------------------------
# App title
# -------------------------------
st.set_page_config(page_title="Cinematic Color Grader", layout="wide")
st.title("🎬 Advanced Cinematic Color Grader")
st.write("Upload an image and apply cinematic color grading using reference images.")

# -------------------------------
# Section 1: Demo Before & After
# -------------------------------
st.subheader("✨ Example Before & After")

before_path = os.path.join("Cover", "before.jpeg")
after_path  = os.path.join("Cover", "after.jpeg")

try:
    before_img = Image.open(before_path)
    after_img = Image.open(after_path)

    st.image(
        [before_img, after_img],
        caption=["Before", "After"],
        use_column_width=True
    )
except Exception as e:
    st.error(f"Could not load demo images: {e}")

    img_before = Image.open(before_path).convert("RGB").resize((400, 400))
    img_after = Image.open(after_path).convert("RGB").resize((400, 400))

    # Create collage (side by side)
    collage = Image.new("RGB", (800, 400))
    collage.paste(img_before, (0, 0))
    collage.paste(img_after, (400, 0))

    # Add overlay text
    draw = ImageDraw.Draw(collage)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((120, 180), "BEFORE", fill=(255, 255, 255), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((540, 180), "AFTER", fill=(255, 255, 255), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))

    st.image(collage, caption="Sample Comparison", use_column_width=True)

except Exception as e:
    st.warning(f"⚠ Could not load demo collage images. Error: {e}")

# -------------------------------
# Section 2: Upload User Image
# -------------------------------
st.subheader("📤 Upload Your Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    src_img = Image.open(uploaded_file).convert("RGB")
    st.image(src_img, caption="Original Image", use_column_width=True)

    # -------------------------------
    # References folder (flat structure)
    # -------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(BASE_DIR, "references")

    if not os.path.exists(ref_dir):
        st.error(f"No 'references/' folder found in {BASE_DIR}")
        st.stop()

    ref_images = [f for f in os.listdir(ref_dir) if f.lower().endswith((".jpg", ".png"))]
    if len(ref_images) == 0:
        st.error("No reference images found in 'references/' folder!")
        st.stop()

    # -------------------------------
    # Section 3: Style Selection
    # -------------------------------
    st.subheader("🎨 Choose Color Grading Style")

    single_style = st.selectbox("Select a single style:", [""] + ref_images)
    blend_styles = st.multiselect("Or blend two styles (max 2):", ref_images, max_selections=2)

    # -------------------------------
    # Section 4: Apply Grading
    # -------------------------------
    if st.button("🚀 Apply Grading"):
        src_np = np.array(src_img).astype(np.float32) / 255.0
        transfer = TransferMVGD()
        result_np = None

        if single_style and not blend_styles:
            ref_img = Image.open(os.path.join(ref_dir, single_style)).convert("RGB")
            ref_np = np.array(ref_img).astype(np.float32) / 255.0
            result_np = transfer.transform(src_np, ref_np)

        elif blend_styles and len(blend_styles) == 2:
            ref_imgs_np = []
            for f in blend_styles:
                ref_img = Image.open(os.path.join(ref_dir, f)).convert("RGB")
                ref_imgs_np.append(np.array(ref_img).astype(np.float32) / 255.0)

            result1 = transfer.transform(src_np, ref_imgs_np[0])
            result2 = transfer.transform(src_np, ref_imgs_np[1])
            result_np = (result1 + result2) / 2.0

        else:
            st.warning("Select either one reference OR exactly two references for blending.")

        # -------------------------------
        # Section 5: Show & Download Results
        # -------------------------------
        if result_np is not None:
            result_np = np.clip(result_np, 0, 1)
            result_img = Image.fromarray((result_np * 255).astype(np.uint8))

            # Enhance cinematic effect
            result_img = ImageEnhance.Contrast(result_img).enhance(1.2)
            result_img = ImageEnhance.Color(result_img).enhance(1.1)

            st.image(result_img, caption="🎬 Color Graded Result", use_column_width=True)

            # Save locally (optional debug)
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", f"{os.path.splitext(uploaded_file.name)[0]}_graded.jpg")
            result_img.save(out_path)
            st.success(f"✅ Saved at {out_path}")

            # ✅ Download button for users
            buf = BytesIO()
            result_img.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download Graded Image",
                data=byte_im,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_graded.jpg",
                mime="image/jpeg"
            )
