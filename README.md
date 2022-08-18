# Neural Style Transfer Web App

link: https://scorpio11rain-nst-web-app-main-nunf6c.streamlitapp.com/
Author: Runyu Tian

## Illustration
The model for the image stylizing leverages pre-trained famous VGG19. All the loss function (content loss, style loss) and gradient descent algorithm are self-implemented. The main hyperparameters to be adjusted are training epochs, content weights and style weights.

In the file neural_style_transfer.py, utils's function for training are defined for stylizing.
In the file main.py, this is the source code for streamlit webpage and image processing pipeline, where you can upload your own content and style image and download your stylized image.

## Notice
Due of lack of GPU and computationally powerful server, the image stylization could take more than 3 minutes. I will continue try to reduce the model's time complexity but the computation limitation is the main reason why the model run slow.




