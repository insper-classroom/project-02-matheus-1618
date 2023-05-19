[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7Wj0oCgF)

# Using Reinforcement Learning for Recommendation Systems

Student: Matheus Oliveira


## Installing
To install all the necessary dependencies of this project, follow the steps above:

* Install [Anaconda](https://www.anaconda.com/download/#linux)

After Anaconda being installed and you've cloned this repo, in your terminal:

```bash
eval "$(/home/<your_user>/anaconda3/bin/conda shell.bash hook)"
export PYTHONPATH=$PYTHONPATH:`pwd`/rl4rs
conda env create -f environment.yml
conda activate rl4rs
```

Download the [data](https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view), and put it in the root folder of this project after extract it.
## Goal
The goal and the scope of this project is to try to use Reinforcement Learning techiniques to train a Recommendations System giving insights to a user input. The implementation of this model will be using [Recsim](https://github.com/google-research/recsim), a Google-associated framework which abstract part of the underlying configuration of the enviroment (using two sources to train an agent).

![image](https://github.com/insper-classroom/project-02-matheus-1618/assets/71362534/9832821a-f53b-489a-a082-e37d05a11172)

## Methods
The methods expected to accomplish this project are:
* Study implemented projects and papers using Recsim;
* Replicate enviroments and projects;
* Use the Gym Wrapper to bring Recsim to a Gym module;
* Try to train a model using tecniques like Q-Learning, DQN or DDQN;
* Compare the results against other Recommendatiom systems implementation such as ML;

## Expected Results 
The expected results are basically:
* Implementation of the framework in Gym like enviroment;
* Comparison in model results and other implementations;
