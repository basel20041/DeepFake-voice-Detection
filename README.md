# Deep Fake-voice-Detection
CNN-Based classification models for detecting Deep Fake Generated vocals using Spectrograms 

--- 
## About the Dataset
- The Dataset link: https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition
## File Structure
- All Plots :
	- this contains all plots of all results obtained while training and validating for all models with all tuning options (according to the approach explained in the notebook)
	- It has about more than 280 plot and many text files
- Fine Tuned Plots:
	- This is the important plots, it contains the accuracy and ROC and confusion matrix of the fine tuned models
- Audio_Deepfake_Tuning.ipynb: the main notebook
- model.py : contains models architecture 
- main.py : a simple flask app to test the models
- torchaudio_feature_extraction.png : this image demonstrates the different ways of representing an audio file in python
- Models Best Weights
	- contains .pth files for each model's best weights
- Final Summery Tuning: a raw text that has each model with its best tuning settings

--- 
# Code Structure
## Notebook :
The Notebook consists of 9 main sections
1. Import Dependencies
	- For this project, several libraries were used but most important was:
		#librosa its a famous Library for audio processing and handling
2. Getting the data ready:
3. Dataset and loaders
4. Modelling (Its also the same for model.py) 
5. Hyper-parameter Tuning — Sequential Coordinate Approach
	- This explains the approach for Tuning the models, and creates training functions for
6. Tuning function
	- Configuration and a function that iterates over all hyper-parameters 
7. Run
8. Summery
9. feature maps plotting 