Project description: 

AI-powered tool for interpreting ECG (electrocardiographs), aimed at identifying Myocardial Infarction and classifying anomaly type across 5 superclasses. 

Model description:

The running model is a reinforced Resnet model – x1resnet1d101 from tsai library: https://timeseriesai.github.io/tsai/

Pipeline:

1. Clone into [https://github.com/alwriza/rayheart.git](https://github.com/alwriza/rayheart.git)  
2. Ensure your machine has all necessary libraries uploaded from the requirements.txt  
3. Run python [dataset.py](http://dataset.py)  
4. Run python [train.py](http://train.py)   
5. Run python predict.py  
   

Setup

pip install torch==2.7.1 tsai==0.4.1 numpy==2.4.3 pandas==3.0.1 wfdb==4.3.1 scikit-learn==1.8.0 matplotlib==3.10.8

Dataset

Download from [https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3](https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3)

