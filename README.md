# **YELP Reviews Sentiment Analysis** 
- [**Dataset**](https://drive.google.com/drive/folders/147VG2_a64juBSJ_hAiB3S8u2fzo4FTC5?usp=sharing)

**Introduction and Objective**
- I have prepared the PyTorch Dataset Class, The Vocabulary Class, The Vectorizer Class, The DataLoader Class, A Perceptron Classifier, The Training Routine, Evaluation, Inference and Inspection Model with the Implementation of PyTorch here in the Project. I have prepared Classifier Model which can predict the Sentiment of a given sentence. I hope you will gain some insights about the Implementation of PyTorch in Natural Language Processing.

**PyTorch Dataset Class**
- PyTorch provides an abstraction for the Dataset by providing a Dataset Class. The Dataset Class is an abstract Operator. When using PyTorch with a new Dataset it is necessary to sub class the Dataset Class and Implement the getitem and len methods.

**The Vocabulary Class**
- The Vocabulary Class not only manages the Bijection i.e Allowing user to add new Tokens and have the Index auto increment but also handles the special token called UNK which stands for Unknown. By using the UNK Token, It will be easy to handle Tokens at Test time that were never seen in Training Instance.

**The Vectorizer Class**
- The second stage of going from a Text Dataset to a vectorized minibatch is to iterate through the Tokens of an Input Data Point and convert each Token to its Integer form. The result of this iteration should be a Vector. Because this Vector will be combined with Vectors from other Data points, there is Constraint that the Vectors produced by the Vectorizer should always have the same length.

**The DataLoader Class**
- The Final step of Text to Vectorized minibatch pipeline is to actually group the Vectorized Datapoints. Because grouping into mini batches is a viatal part of Training the Neural Networks, PyTorch provides a built in class called DataLoader for coordinating the Process.

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re, json, string, os
import collections

from argparse import Namespace
from IPython.display import display 
from collections import Counter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
```

**Getting the Data**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I have used [**Yelp Reviews Dataset**](https://www.kaggle.com/yelp-dataset/yelp-dataset) for this Project. In 2015, Yelp held a contest to predict the Rating of the Restaurants given it's Reviews. Zhang, Zhao, and Lecun simplified the Dataset by converting the Ratings into Sentiments viz. Positive Sentiment for 3 to 4 star Ratings and Negative Sentiment for 1 to 2 star Ratings. The Dataset is splitted into 560,000 Training Samples and 38,000 Testing Samples.

**Processing the Data**
- I have presented some Data Preprocessing Techniques such as Cleaning the Dataset and Creating Training, Validation and Test Splits from the DataFrame which I have Implemented while working with YELP Reviews Dataset for Sentiment Analysis here in the Snapshot. 

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2036.PNG)

**PyTorch Dataset Class**
- PyTorch provides an abstraction for the Dataset by providing a Dataset Class. The Dataset Class is an abstract Operator. When using PyTorch with a new Dataset it is necessary to sub class the Dataset Class and Implement the getitem and len methods. I have presented the Implementation of Review Dataset Class using PyTorch here in the Snapshots.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2037a.PNG)

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2037b.PNG)

**Preparing the Classifier Model**
- I have presented the simple Implementation of PyTorch in Training the Classifier Model along with the process of Instantiating the Dataset, Model, Loss, Optimizer and Training State here in the Snapshots.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2038b.PNG)

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2038a.PNG)
