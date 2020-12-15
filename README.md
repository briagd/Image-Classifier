# Image-Classifier

Image classifier using pretrained Resnext and Resnet models using an ensemble with soft voting.
The dataset used contains 2827 image samples, and was used in the ImageCLEF 2008 competition. 1827 were used for training and validation and 1000 for testing.

The model produce a micro precision of 0.83, micro recall of 0.84 and micro F1 score of 0.84 on the test data.

The training/validation split can be found in ```train.csv``` and ```val.csv``` 


To train the models, edit the path of the training data in the ```train``` function in ```__init__.py``` file, then navigate to the root folder of the project and run:
```python 
python -m classifier -train
```

To evaluate the model on the validation, edit the path of the validation data in the ```evaluate_validation``` function in ```__init__.py``` file, then navigate to the root folder of the project and run:
```python 
python -m classifier -eval_val
```

To evaluate the model on the test data, edit the path of the test data in the ```evaluate_test``` function in ```__init__.py``` file, then navigate to the root folder of the project and run:
```python 
python -m classifier -eval_test
```
