# kaggle-LANL-Earthquake-Prediction
Kaggle Conquest https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview

With acoustic signal data we try to predict next earthquake. ( This was a team project, we used different models but at the rest of summary I will explain my model, which has a highest score.)

For feature extraction we use both raw signal data and Fast Fourier Transform of raw signal. Then we give these features to different two Neural Network. After that we concatenate outputs of NNs and use as input for one last NN. 

We taught using FFT could give us some clues about frequency of earthquakes. But most of features from FFT signal found useless (just a few of them impact the results).

Our model ranked as 189th between 4540 models.
