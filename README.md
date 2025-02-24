first step -
torch: The core PyTorch library for deep learning.
torch.nn: Contains classes for building neural networks.
torch.optim: Includes optimizers like Adam, SGD, etc., for training.
torchvision: A PyTorch module for handling image datasets and pre-trained models.
torchvision.transforms: Helps preprocess and augment images before training.
2nd step- 
load gpu
3rd step - 
we download cifar data set, CIFAR-10 is a dataset containing 60,000 color images of size 32x32 across 10 classes (airplane, car, bird, cat, etc.).
transforms.Compose([...]): This creates a sequence of transformations that will be applied to the images in the dataset.
transforms.ToTensor(): This converts the images (which are typically PIL Image or NumPy arrays) into PyTorch tensors. 
Tensors are the fundamental data structure used in PyTorch for numerical computation. 
This transformation also rearranges the image data from a shape of (height, width, channels) to (channels, height, width), and it scales the pixel values from the range [0, 255] to [0.0, 1.0].
4th step - 
Defining the Colorization Model - nn.Module is the base class for all neural network modules in PyTorch.
