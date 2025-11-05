# assignment--5

Objective:
To build a deep learning model that can automatically detect Tuberculosis (TB) from chest X-ray images, improving early diagnosis and assisting healthcare professionals.

Model Architecture (CNN):

Input Layer: Takes resized X-ray images.

Convolutional Layers: Extract features like edges and textures.

Activation Function (ReLU): Introduces non-linearity.

Pooling Layers (Max Pooling): Reduce spatial size, keep important features.

Flatten Layer: Convert feature maps to 1D vector.

Dense Layers (Fully Connected): Combine features for classification.

Output Layer:

Binary Classification: 
Sigmoid (TB vs Normal).
