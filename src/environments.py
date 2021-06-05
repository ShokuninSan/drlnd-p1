class UnityEnvWrapper:
    
    def __init__(self, unity_env):
        self.env = unity_env
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = None
        
    def reset(self, train_mode=True):
        self.env_info = self.env.reset(train_mode)[self.brain_name]
        return self.env_info.vector_observations[0]
    
    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done
    
    def close(self):
        self.env.close()