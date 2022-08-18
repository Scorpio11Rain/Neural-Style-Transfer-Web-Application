import streamlit as st
import neural_style_transfer
import cv2

ACCEPT_IMG_TYPES = ["jpeg", "png", "jpg"]

st.markdown('Author: Runyu Tian')
st.title('Neural Style Transfer App')
col1, col2 = st.columns(2)

content_image = st.sidebar.file_uploader("Load your content image below:", type = ACCEPT_IMG_TYPES, accept_multiple_files = False, key = None, help = "Upload your content image that you want to stylize!")

style_image = st.sidebar.file_uploader("Load your style image below:", type = ACCEPT_IMG_TYPES, accept_multiple_files = False, key = None, help = "Upload your style image that you want to stylize on the content image!")

with st.spinner("Loading content image..."):
    if content_image is not None:
        col1.subheader("Your content image")
        col1.image(content_image,use_column_width = True)
        raw_img,raw_shape = neural_style_transfer.load_img_byte(content_image)

    else:
        col1.text("Image not uploaded yet")
        col1.text("This's sample illustration")
        col1.text("Please upload images!")
        col1.image("example_img/cat.jpg")
        

with st.spinner("Loading style image"):
    if style_image is not None:
        col2.subheader("Your style image")
        col2.image(style_image,use_column_width = True)
        style, style_shape = neural_style_transfer.load_img_byte(style_image)
    else:
        col2.text(" ")
        col2.text(" ")
        col2.text(" ")
        col2.image("example_img/van.jpg")

clicked = st.sidebar.button("Start your own stylization") and content_image and style_image

if clicked:
    vgg = neural_style_transfer.load_vgg_model()
    merged = neural_style_transfer.stylize(raw_img, style, vgg, 50)
    merged = cv2.resize(merged, raw_shape)
    st.subheader("Your stylized image")
    st.image(merged)
    st.download_button(label="Download Final Image", data=neural_style_transfer.save_image(merged, raw_shape), file_name="stylized_image.png", mime="image/png")

else:
    st.subheader("Example ouputs from above")
    st.text("Please run button on the left to have your own stylized image!")
    st.image("example_img/merged.jpg")  
