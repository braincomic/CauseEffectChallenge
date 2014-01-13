CauseEffectChallenge
====================

My submission to the ChaLearn cause-effect pair challenge, which was hosted by Kaggle

Version: 1.0

Date: 19 August 2013

Environment: Python 2.7.2

OS: Windows XP

Computer: Intel(R) Xeon(R) CPU, E-3-1230 V2 @ 3.30GHz, 3.29 GHz, 3.47 GB of RAM

Disk space needed: limited (less than 1 GB)

Time needed to predict: about 2 hours

Best score obtained from this version: 0.79026


=== Installation instructions ===

This is simply an extension of the Python benchmark, provided by Kaggle and ChaLearn.

1. Load libraries to Python. 
   The only libraries that are used are very 'typical' libraries for data scientists,
   so they are probably already installed.
   * numpy
   * pandas
   * scipy
   * sklearn
   * pylab (optional, might give import warnings/errors)
   
2. Modify SETTINGS.json with the correct locations of the input and output files.

3. In the source code of ce.py, update the folder in which the script and SETTINGS.json is located:
   * locationFolder = "C:/Kaggle/KaggleCauseEffect/PYTHON/"
   
4. Run the program to make the predictions. In python, import ce.py and call the predictAll() method.
   (or simply run the program, because predictAll() is in the main function).

   from ce import *
   train()
   predictAll()
   
   The last line on the screen is "Finished predicting".
   
   This created the three possible predictions: predictions_full.csv,
   predictions_both.csv and predictions_weighted.csv

