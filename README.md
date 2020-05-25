# tennis_AI

# Introduction
In the past my friends and I have always bet on the outcome of tennis matches, and had competitions to see who could fill out the most correct bracket before tournaments. I want to see how different machine learning methods, particularly random forests and neural networks could be used to predict this.

# Data
The data used is from https://github.com/JeffSackmann/tennis_atp.
It has data stretching from the 1960's to present day on ATP matches, and accompanying variables with each match.

# Feature Engineering

From what I have seen online, when people use machine learning to predict sports matches they will frequently state results such as a validation accuracy of 65%. But in my mind this is not very good if just predicting the favourite everytime will also give you accuracy of 65%.
I set up my network so that there would be one group of features for the "overdog" (favourite), and another group of features for the "underdog". Then the Y variable would be 0 if the overdog won, and 1 if the underdog won. This way I would be able to easily how much better my neural network was performing compared to just predicting the overdog everytime. This was done through underdog_overdog.py.

I then engineering the following features

## Times Upset Recently

Given that we are trying to predict the possibility of an upset here, it makes sense to engineer features to capture this probability. One is the number of times that the overdog has been upset recently, and another is the number of times that the underdog performed and upset recently.

## Previous Head2Head

The previous head2head of players is crucial. An underdog may have a particularly good record against an overdog

## Previous Stats

Also include the average of the recorded stats for the past ten matches (e.g. first serve percentage, break points faced, etc.)

## Recent Win/Loss
This will give a better picture of how hot the player is. Four features in total: underdog_recent_loss, overdog_recent_loss,underdog_recent_win,overdog_recent_win
Also added in four surface specific recent win/loss metrics. 

# Initial Neural Network Training
train.py
Split the data up so that the test set was just data from 2019-2020. This will be the best way to judge the models ability to predict future outcomes given previous trianing data.
Training was done using cross validation in Pytorch. Experiments were tracked using wandb.

![alt text](https://github.com/PeterBHull/tennis_AI/blob/master/initial_training.jpg?raw=true)


