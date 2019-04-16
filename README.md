# Google Landmark Retrieval 2019 Competition fast.ai Starter Pack

The code here is all you need to do the first submission to the [Google Landmark Retrieval 2019 Competition](https://www.kaggle.com/c/landmark-retrieval-2019). It is based on [FastAi library](https://github.com/fastai/fastai) release 1.0.47 and borrows helpser code from great [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) library. The latter gives much better results than code in the repo, but not ready-to-make submission and takes 3 days to converge compared to 45 min here. 


## Making first submission
1. Install the [fastai library](https://github.com/fastai/fastai), specifically version 1.0.47. 

2. Install the [faiss library](https://github.com/facebookresearch/faiss). 
conda install faiss-gpu cudatoolkit=9.0 -c pytorch-y

3. Clone this repository. 

4. Start the download process for the data. It would take a lot, so in mean time you can run the code.

5. Because the code here does not depend on competition data for training, only for submission. 

## Notebooks

1. [download-and-create-microtrain](https://github.com/ducha-aiki/google-retrieval-challenge-2019-fastai-starter/blob/master/download-and-create-microtrain.ipynb) - download all the aux data for training and validation
2. [validation-no-training](https://github.com/ducha-aiki/google-retrieval-challenge-2019-fastai-starter/blob/master/validation-no-training.ipynb) - playing with pretrained networks and setting up validation procedure
3. [training-validate](https://github.com/ducha-aiki/google-retrieval-challenge-2019-fastai-starter/blob/master/training-validate.ipynb) - training DenseNet121 on created micro-train in 45 min and playing with post-processing
4. [training-validation](https://github.com/ducha-aiki/google-retrieval-challenge-2019-fastai-starter/blob/master/training-validate.ipynb) - creating a first submission. Warning, this could take a lot (~4-12 hours) because of the dataset size
