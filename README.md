[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7Wj0oCgF)

# Using Reinforcement Learning for Recommendation Systems

Student: Matheus Oliveira


## Installing
```bash
pip install -r requirements.txt
```
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

## Expected Results 
The expected results are basically:
* Hit and DCG metrics;
* Python Script to recommend movies due randomic input;
* Analisys and conclusions about RL for Recsys.