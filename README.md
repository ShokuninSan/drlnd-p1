[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

In this project we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Project Layout

The following project tree shows where to find code, documentation, etc.

```
.
├── README.md ... this README
├── environments
│   └── ... your Unity environment goes here
├── models
│   └── drlnd_p1_model.pth ... the serialized (trained) model weights
├── notebooks
│   ├── Navigation.ipynb ... the entry point where you can train and/or test the agent
│   ├── Report.ipynb ... the project report
│   └── scores.png ... saved plot (shows average rewards of episodes)
├── python
│   └── ... contains and defines project depedencies (mostly borrowed from https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
├── src
│   ├── agents.py ... contains the DoubleDDQN agent implementation
│   ├── environments.py ... contains wrapper for the Unity env
│   ├── experiences.py ... contains replay buffers
│   └── models.py ... contains neural network implementations
```

### Project Details

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Project Setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the `environments/` folder, and unzip (or decompress) the file. 

3. [Optional] Create a Conda environment and activate it
```
(base) ➜  drlnd-p1 git:(master) ✗ conda create --name drlnd-p1 python=3.6
(base) ➜  drlnd-p1 git:(master) ✗ conda activate drlnd-p1
```

4. Change into the `python` folder and execute `pip install .` to install the required dependencies.

5. Create a custom IPython kernel by executing `$ python -m ipykernel install --user --name drlnd --display-name "drlnd"`

### Getting Started

Start a `jupyter notebook` from within the project folder and follow the instructions in `notebooks/Navigation.ipynb` to either
* train your own agent or
* load the model weights and watch the pre-trained agent

__HINT__: make sure to switch from the default Python 3 kernel to "drlnd" (see section Project Setup).

---

Tested on macOS Big Sur (Version 11.0.1) and Ubuntu 20.04.2 LTS.