# neural_cotraining: The pipeline for performing MTL on object recognition with neural data prediction

![Black](https://github.com/sinzlab/nnfabrik/workflows/Black/badge.svg)
![GitHub Pages](https://github.com/sinzlab/nnfabrik/workflows/GitHub%20Pages/badge.svg?branch=master)

neural_cotraining is an MTL pipeline, mainly developed for cotraining neural networks on image classification alognside neural prediction. It is mainly built on [nnfabrik](https://github.com/sinzlab/nnfabrik), [neuralpredictors](https://github.com/sinzlab/neuralpredictors), [nntransfer](https://github.com/sinzlab/nntransfer), [nnvision](https://github.com/sinzlab/nnvision) and [bias_transfer_recipes](https://github.com/sinzlab/bias_transfer_recipes).

## Why use it?

Co-training a neural network on two tasks can have multiple challenges such as:
- Having multiple datasets to iterate over per epoch
- Usage of a different loss function for each task
- Training and testing on different sets basides the logging of all the results.

Our framework provides a flexible way to run experiments related to MTL that is performed on image classification alongside neural data prediction. We provide a flexible way to alternate between batches from the different datasets in addition to manually controlling the number of batches for each dataset. Besides, it is possible to weigh the different losses that is learnable throughout likelihood losses. Furthermore, we handle the logging and storing of results per epoch in a simple and efficient way.

## :gear: Installation

You can use the following way to install neural_cotraining:

#### 1. Via GitHub:
```
pip install git+https://github.com/sinzlab/neural_cotraining.git
```

## Structure of the code 

Based on nnfabrik and neuralpredictors, the pipeline consists of 3 major components: dataset (loader), model (builder) and the (co-)trainer.
### Dataset
In the "/configs" folder can be found the dataset configs which are used for different tasks: image classification, neural prediction or MTL on both. Also, the loaders' functions can be found in the "/dataset" folder.
### Model
In the "/configs" folder can be found the model configs which are used for different tasks: image classification, neural prediction or MTL on both. Also, the model-building functions can be found in the "/models" folder.
### Trainer
In the "/configs" folder can be found the trainer config which can be used for MTL. Also, the trainer function can be found in the "/trainer" folder.

## :bulb: Example

Based on the above mentioned libraries, we provide a simple example on how to run an MTL experiment using neural_cotraining. 
[Here](./examples/notebooks/nnfabrik_example.ipynb)


## :bug: Report bugs (or request features)

In case you find a bug, please create an issue or contact any of the contributors.
