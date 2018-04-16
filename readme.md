# Meta Bandits
### Apr 15, 2018

## Description
Minimal recreation of the Botvinick paper [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763) using A2C.

This project seeks to train an RNN that posesses the ability to learn without plasticity (without changing the weights). The approach is an artificial simulation of an experiment done with monkeys. 

The experiment consisted of a monkey given the choice of uncovering one of two different colored cups. Beneath one of the two cups is a reward in the form of some tastey food. An episode within the experiment consisted of 10 trials. Each trial consisted of the same monkey allowed to pick between the two choices. For the entire trial, the food was always located under the same colored cup. Between each trial one of the two colored cups was chosen as the new color to have the food under it. 

The results were that the monkeys learned to consistenly pick the reward giving cup after the first step in the trial (in which the monkey had a 50-50 chance of selecting the reward cup). 

This is the experiment that this project seeks to emulate with a Recurrent Neural Network. The goal is to create an RNN that can leverage its past experience to pick the reward giving cup with minimal regret.

## Experimental Setup
For every step in a trial, the RNN is made to select one of 2 cups. One of the two cups provides a +1 reward with probability p. The other cup has probability (1-p) of containing the food. At every step, one of the two cups is guaranteed to provide a +1 reward. The other provides a reward of 0. The input to the RNN is the action taken at the last step along with the reward received at the last step. The hidden state is not reset until the end of a trial. A GRU is the RNN used for the experiment.
