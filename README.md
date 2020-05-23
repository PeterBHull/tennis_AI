# tennis_AI

# Introduction
In the past my friends and I have always bet on the outcome of tennis matches, and had competitions to see who could fill out the most correct bracket before tournaments. I want to see how different machine learning methods, particularly random forests and neural networks could be used to predict this.

# Data
The data used is from https://github.com/JeffSackmann/tennis_atp.
It has data stretching from the 1960's to present day on ATP matches, and accompanying variables with each match.

# Feature Engineering

From what I have seen online, when people use machine learning to predict sports matches they will frequently state results such as a validation accuracy of 65%. But in my mind this is not very good if just predicting the favourite everytime will also give you accuracy of 65%.
I set up my network so that there would be one group of features for the "overdog" (favourite), and another group of features for the "underdog". Then the Y variable would be 0 if the overdog won, and 1 if the underdog won. This way I would be able to easily how much better my neural network was performing compared to just predicting the overdog everytime.

