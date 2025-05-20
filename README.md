# VRDL-Final-Project

## dataset
if your are CP Lab member
```
cp /project/YearGuessr/cassava-leaf-disease-classification.zip ./
mkdir data
unzip cassava-leaf-disease-classification.zip -d ./data
```
else, please download the dataset at [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
```
kaggle competitions download -c cassava-leaf-disease-classification
mkdir data
unzip cassava-leaf-disease-classification.zip -d ./data
```

## Val Accuracy

* convnext: 0.8850
* bioclip:  0.8963