# DDCCNet_public
DDCCNet models are neural network (NN) models tasked with predicting t<sub>1</sub> and t<sub>2</sub> amplitudes for coupled-cluster singles and doubles (CCSD) calculations.
We (Vogiatzis group at University of Tennessee, Knoxville) have developed three versions of these models,
1. DDCCNet_v1
2. DDCCNet_v2
3. DDCCNet_v3

   and these models use MP2 level electronic structure data as input in predicting the CCSD amplitudes.

Following is the general workflow of the DDCCNet models.
1. Generate Data for Training
2. Training the models
3. Data generation for testing
4. Testing the models

Table below shows the files associated with each step of the workflow for each DDCCNet version.

|Function                  |[DDCCNet_v1](DDCCNet_v1)            |[DDCCNet_v2](DDCCNet_v2)           |[DDCCNet_v3](DDCCNet_v3)           |
|--------------------------|----------------------|---------------------|---------------------|
|1. Generate Data for Training |[Data_Generation_v1.py](DDCCNet_v1/Data_Generation_v1.py)|[Data_Generation_v2.py](DDCCNet_v2/Data_Generation_v2.py)|[Data_Generation_v3.py](DDCCNet_v3/Data_Generation_v3.py)|
|2. Training the models        |[Train_NN_v1.py](DDCCNet_v1/Train_NN_v1.py)       |[Train_NN_v2.py](DDCCNet_v2/Train_NN_v2.py)       |[Train_NN_v3.py](DDCCNet_v3/Train_NN_v3.py)       |
|3. Data generation for testing|[Test_NN_v1.py](DDCCNet_v1/Test_NN_v1.py)        |[Save_test_feats_v2.py](DDCCNet_v2/Save_test_feats_v2.py)|[Test_NN_v3.py](DDCCNet_v3/Test_NN_v3.py)        |
|4. Testing the models         |[Test_NN_v1.py](DDCCNet_v1/Test_NN_v1.py)        |[Test_NN_v2](DDCCNet_v2/Test_NN_v2.py)           |[Test_NN_v3.py](DDCCNet_v3/Test_NN_v3.py)        |
