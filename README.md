## IT-Talents Big Data Predictions Competition 2018

This analysis was created by Robin Denz and submitted to the Big Data Predictions Competition on 31.12.2018.
Further information about the competition as well as the dataset can be found at: https://www.it-talents.de/foerderung/code-competition/code-competition-12-2018

In the competition, participants were tasked with analyzing the supplied data in any way they like. The goal was to ask interesting questions the data might be able to answer, build predictive models and visualize your results.

The code in this respiratory is my take on that challenge.

# Files:

Denz_Analysis.ipynb -- This is the jupyter notebook containing the complete analysis.

Denz_Analysis.py    -- This is a python script version of the analysis. It contains nearly the same code, but it's less "clean".

donut_outcome_percentages.png -- Graph of outcome percentages before removing unfinished games.

donut_outcome_percentages2.png -- Graph of outcome percentages after removing unfinished games.

target_correlations.png -- Graph of correlations with the target variable.

previous_win_percent_in_matchup.png -- Lowess regression of previous win percentage and target variable.

previous_win_percentage_difference.png -- Lowess regression of previous win percentages difference and target variable.

feature_importances_base.png -- Feature importances plot of the simplest random forest model.

feature_importances_final.png -- Feature importances plot of the final random forest model.

correlation_heatmap.png -- Heatmap plot of all variable correlations.

Cause of copyright reasons I may not upload the datset here. You may however download it for free on the competitions official website (https://www.it-talents.de/foerderung/code-competition/code-competition-12-2018).

# How to View the Project

You can download the ".py" file and the corresponding .png files to take a look at the analysis without actually rerunning the code yourself.

If you actually want to rerun the project, you will first need to download all files from this respiratory and the dataset ("races.csv") from the official website.

To run the jupyter notebook, open jupyter with a command prompt and click on the file.

Example in Windows PowerShell:

cd my_file_path
jupyter notebook

# Prerequisites

The entire code is written in python 3.6, so you will need this installed on your computer to rerun the code.
Any official python version may be downloaded here: https://www.python.org/downloads/

In addition, you will need to install the following packages:

jupyter
scikit-learn
scipy
numpy
calendar
warnings
pandas
seaborn
matplotlib
statsmodels

You can install those using the pip install method. For more information, please refer to their respective official documentation.
