[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7Wj0oCgF)

# Using Reinforcement Learning for Recommendation Systems

Student: Matheus Oliveira


## Installing
Create an virtual enviroment and install the dependencies of the project:

```bash
pip install -r requirements.txt
```

## Files distribution
* recsys_rl is the notebook where the agent were trained using the article and **shashist** implementation;
* recommentation.py is a file that uses a trained agent to recommend three movies due input of three random movies by the user (usually are related recommendations). Use it as:
```bash
python recommendation.py
```
* utils.py and classes.py are file helpers with Classes and functions that are using in the recommendation script.


## Goal
The goal and the scope of this project is to try to use Reinforcement Learning techiniques to train a Recommendations System giving insights to a user input. The implementation of this model will be using different sources to try to accomplish a RL agent that can Recommend movies based in the Dataset. 

This is a re-implementation of [RecSys-RL](https://github.com/shashist/recsys-rl) and based on [Feng Liu](https://arxiv.org/pdf/1810.12027.pdf) article.

## Methods
The methods expected to accomplish this project are:
* Study implemented projects and papers using RL in Recommendation Systems;
* Replicate enviroments and projects;
* Train the model;
* Create a python Script to recommend three movies based in three random initial movies picked by random;

This model is using a DDPG model behind Actor-Critic Archictecture.
![image](https://github.com/insper-classroom/project-02-matheus-1618/assets/71362534/bbee6538-b924-41e3-b8b4-3034f39ce28c)

The general overview of the implemetion follow the Schema below:
![image](https://github.com/insper-classroom/project-02-matheus-1618/assets/71362534/a6d4f690-ef61-4187-8ffd-3b9180fbcd6b)


## Expected Results 
The expected results are basically:
* Hit and DCG metrics;
* Python Script to recommend movies due randomic input;
* Analisys and conclusions about RL for Recsys.
