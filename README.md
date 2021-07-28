# neural_cotraining: The pipeline for performing MTL on object recognition with neural data prediction


neural_cotraining is the code for our paper [Towards robust vision by multi-task learning on monkey visual cortex](). Here we implement an MTL pipeline, mainly developed for cotraining neural networks on image classification alognside neural prediction. It is mainly built on [nnfabrik](https://github.com/sinzlab/nnfabrik), [neuralpredictors](https://github.com/sinzlab/neuralpredictors), [nntransfer](https://github.com/sinzlab/nntransfer), [nnvision](https://github.com/sinzlab/nnvision) and [bias_transfer_recipes](https://github.com/sinzlab/bias_transfer_recipes).

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

## Code 

Based on nnfabrik and neuralpredictors, the pipeline consists of 3 major components: dataset (loader), model (builder) and the (co-)trainer. In the `/configs` folder can be found the config classes for the different components which are relevant for image classification, neural prediction and MTL on both tasks.
### Dataset
The dataset loaders' functions can be found in the `/dataset` folder. The function in `/dataset/neural_dataset_loader.py` can load the monkey V1 dataset or a similar neural dataset through the `monkey_static_loader` in `/nnvision/datasets/monkey_loaders.py` using the `neural_cotrain_NeurIPS` branch. Furthermore, the function in `/dataset/img_classification_loader.py` can load an image classification-related dataset like TinyImageNet directly.
In order to combine both loaders in the MTL setup, we use the `MTLDatasetsLoader` in `/dataset/mtl_datasets_loader.py`.

### Model
The model-building functions can be found in the `/models` folder. The builder function in `/models/vgg.py` can create a VGG model to perform single-task image classification. Also, the builder in `/models/neural_model_builder.py` creates a standard model to predict neural responses using the models implemented in nnvision. 
To combine image classification and neural prediction in one model, we use the `MTL_VGG` model class in `/models/mtl_vgg.py`. 

This is the architecture of our co-trained monkey model using the MTL_VGG class ![](https://github.com/Shahdsaf/neural_cotraining/blob/main/mtl_vgg.png)

### Trainer
In order to train the defined model, we use the trainer function, that can be found in the `/trainer` folder. It is responsible for the whole training process including the batch iterations, loss computation, evaluation on the validation set and the logging of results per epoch and finally the final evaluation on the test sets.

## :bulb: Example

In order to run experiments and populate them in the datajoint tables, we use the bias_transfer_recipes library, which helps to specify the experiment's parameters in a simple manner by defining the attributes of the dataset, model and trainer configs.
We provide a simple example on how to run an MTL experiment using neural_cotraining 
[Here](https://github.com/Shahdsaf/bias_transfer_recipes/blob/uptodate_shahd/bias_transfer_recipes/notebooks/example.ipynb). In addition, we provide ready-to-use scripts to reproduce the results or model weights we used in our paper following this [link](https://github.com/Shahdsaf/bias_transfer_recipes/tree/uptodate_shahd/bias_transfer_recipes/notebooks).


## :bug: Report bugs 

In case you find a bug, please create an issue or contact any of the contributors.
