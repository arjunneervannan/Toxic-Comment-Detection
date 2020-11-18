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

This project was completed between October of 2018 and January 2020 (with majority of the code written between November 2018 and February 2019).

Some links to press releases of this project:
https://www.societyforscience.org/regeneron-sts/2020-finalists/
https://newsroom.regeneron.com/news-releases/news-release-details/teen-scientists-win-18-million-virtual-regeneron-science-talent
https://parentology.com/meet-2-of-the-regeneron-science-talent-search-2020-finalists-changing-the-world/
https://fisher.wharton.upenn.edu/mt-for-life/arjun-neervannan-mt-24-on-combating-cyberbullying-with-ai-and-winning-one-of-the-regeneron-sts-scholarship-prizes/
