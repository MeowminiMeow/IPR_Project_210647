## Project Overview
This project implements the model from the paper [**DWFORMER: Dynamic Window Transformer for Speech Emotion Recognition**](https://arxiv.org/abs/2303.01694) by Shuaiqi Chen, Xiaofen Xing, Weibin Zhang, Weidong Chen, and Xiangmin Xu. The goal of the project is to perform **Speech Emotion Recognition** using the **IEMOCAP Dataset**.

Download the IEMOCAP Dataset from drive link attached. 
We will have to launch two seperate python environment 
1) One for feature extraction
2) Another for model training and validation
   
This is done to prevent any issues rising up as few of the modules use different version of libaries which can cause issue in installing the **Fairseq** module by Facebook. 

## Dataset
The **IEMOCAP Dataset** is used in this project. The original paper also uses **MELD Dataset** ,however as stated in this conversation, the author has used a different method of pre-processing the code and has not uploaded that, as stated in this [github conversation](https://github.com/scutcsq/DWFormer/issues/17), apart from that **MELD Dataset** also has some error which is even stated in the official website. 

## Environment Setup

Firstly, Make a folder named **downloads** use this to save all the external downloads ,over here. 

This environment is for the usage of **WaveLM** to extract the features from th audio files. 
```bash
conda create -n python_without_fairseq python=3.9
conda activate python_without_fairseq
pip install pip==24.0
pip install ipykernal

!conda install -y gdown
!gdown --id 12-cB34qCTvByWT-QtOcZaqwwO21FLSqU ## installs the waveLM.pt
pip install pandas
pip install matplotlib
pip install einops
pip install lmdb
pip install librosa
```
The second environment is will be used to install the **Fairseq** module. 
```bash
conda create -n python_with_fairseq python=3.8
conda activate python_with_fairseq
pip install ipykernal

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
##Install Fairseq lib
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
## How to run this code

### 1. Clone the repository
Begin with cloning this project report
```bash
conda activate python_without_fairseq
git clone https://github.com/username/IPR_Project_210647.git
cd IPR_Project_210647
```
### 2. Developing the features
Open the **Feature_Extractor** folder,Open the **data_preprocess.py** and then make sure to do the following changes:-
1) **Line 18** : Copy the path of the location of WavLM-Large.pt and then change it at the line checkpoint.
2) **Line 27** : In the **iemocap_data** folder, copy the path of any one of the datafile and change it.
3) **Line 28 and 29** : Change the line 28 to the location of the download of the data and Line 29 to the place where we want to save this processed dataset.

Once all these changes are made, run the following command: 
```bash
python ./data_preprocess.py
```
### 3. Dataset Labeling
Open the **Dataset.py** in IEMOCAP Folder , make the following changes:
1) **Line 45** : Change this to the location where you want to save in, this will save the files in the format of **data.mdb** and **lock.mdb**
2) **Line 48 and 56** : This stores the place where the **train.csv** and **valid.csv** files are stores in the Feature Extractor folder.
3) **Line 49** : The output directory of the pervious feature extraction is input here , this will take the features and label them accordingly.
Once all these changes are made, run the following command: 
```bash
python ./Dataset.py
```

### 4. Model Training

In IEMOCAP Folder, Open the **train.py** file , and do the following changes:
1) **Line 71** : Add the location where the data features are saved, this is the output of the previous step
2) **Line 131 and 178** : Input the location of the CSV file to be saved , this csv file collects the session wise weighted accuracy and unweighted accuracy, including loss and time taken to train for each step.

Once all these changes are made, run the following command: 
```bash
python ./train.py
```
### 5. Evaluation
Using the code in the **IPR_Project_1.ipynb** to get the graph needed to compare with the original paper. 

In paper uses valid session scores to measure the result of the model. 
We have achieved a weighted accuracy of **0.7192332746081762** compared to that of the paper of **0.723** and when compared to unweighted accuracy , we have achieved **0.7313218642765257** compared to that of the paper of **0.739**. Thus our error is less than **1%** of the actual results. 
 


