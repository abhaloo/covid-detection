# Covid-Detection Classifier

### Description:
A neural network used to classify CT scans of COVID-19 patients. This neural network is based on the VGG16 architecture which is still considered good for vision architecture.
The network is quite large and has millions of parameters among all the layers and hidden layers. 

### How to run:
1. This project can be imported to Google Colab to run the notebook.   
2. On Google Colab, make sure to change runtime to "GPU" for accelerated performance.   
3. Provide images for the model to train and validate on by clicking on the folder on the left hand side of the screen.   
    Right-click to upload a new folder called "data". The structure of directories within data must be maintained so that they can be read inside the notebook.   
4. Run each cell going from the top of the notebook to the bottom.   

### Useful Resources:
Data Source: https://github.com/UCSD-AI4H/COVID-CT  
VGG16 Implementation: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py  
Tutorial: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c  
Tutorial: https://github.com/appushona123/Instant-COVID19-detection/blob/master/covid19_hackathon.ipynb  
