# Toxic-Comment-Detection

This repository contains the code for the project "Combating Cyberbullying and Toxicity by Teaching AI to Use Linguistic Insights from Human Interactions in Social Media", which won the 10th place prize at the Regeneron STS Competition, the oldest and most prestigious science fair in the US.

The goal of this project was to develop a novel method to detect and mitigate bias in an toxic comment detection algorithm. There were a couple steps to this:

1. Preprocessing data (completed in data.py)
2. Building the initial baseline model (completed in model.py)
3. Training the model on 100k comments, and testing it (completed in main.py)
4. Detecting biases by using attention weights to find words that model was most biased against (completed in main.py)
5. Augmenting the training dataset with these words, grid-searching over number of comments per words and number of words, and retraining for each of these steps (completed in main.py)
6. Picking the best model out of these (based on a metric that measures percent change in False Positive Equality Difference; completed in main.py)
7. Repeating all trials 5 times to minimize training variance.

Server.py includes a script to run a live demo of the model at http://detoxifai.com.
