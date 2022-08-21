# Neural Style Transfer Web App

link: https://scorpio11rain-nst-web-app-main-m9dkk2.streamlitapp.com/
Author: Runyu Tian

## Screenshot of web app page below
![Xnip2022-08-18_10-30-56](https://user-images.githubusercontent.com/88039791/185458456-893d1a1a-37aa-4b04-98fe-3799232b0744.jpg)

## Illustration
The model for the image stylizing leverages pre-trained famous VGG19. All the loss function (content loss, style loss) and gradient descent algorithm are self-implemented. The main hyperparameters to be adjusted are training epochs, content weights and style weights.

In the file neural_style_transfer.py, utils's function for training are defined for stylizing.
In the file main.py, this is the source code for streamlit webpage and image processing pipeline, where you can upload your own content and style image and download your stylized image.

## Notice
Due of lack of GPU and computationally powerful server, the image stylization could take more than 3 minutes. Also, the stylization effect might not be that obvious in some cases, this is due to low number of training epochs. It is currently only set to be 40 to have obvious stylization effect, please adjust it to more than 1000, but this might take really a long time with GPU acceleration.

In order to tune your own epochs, please find it in the "EPOCHS" global variables in main.py and tune it to be your prefered number.



