class myDQAgent():
    
    def __init__(self, model, env, epsilon=0.2, epsilon_lim=0.05, 
                 gamma=0.5, decay=0.9, batch_size=32, memory_limit=2000, replay=False):
        # Class for deep q agent. Takes a model, an environment and optional hyperparameters
        # to it's constructor
        self.model = model
        self.env = env
        self.epsilon = epsilon # 1-epsilon chance of random action choices during training
        self.epsilon_limit = epsilon_lim # a minimum bound on epsilon that decay cant go below
        self.discount = gamma # discount factor
        self.decay = decay # decay is applied to epsilon after each training round
        self.batch_size = batch_size 
        self.replay=replay # Boolean determines if training is done with or without memory buffer
        self.memory_limit = memory_limit # a limit on the size of the memory buffer
        self.memory = self.init_memory(self.memory_limit, self.batch_size, self.env)
        
    def init_memory(self, limit, batch_size, env):
        mem = deque(maxlen=limit)
        return mem

    def getActionValue(self, state):
        return np.max(self.model.predict(np.expand_dims(state, axis=0)))

    def getAction(self, state, rand=True):
        if rand == False: # Non randomized action choices
            pred = self.model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(pred)
            return action
        else:
            if np.random.random() < self.epsilon: # action choices in accordance with epsilon hyperparameter
                return np.random.randint(0, self.env.action_space.n)
            else:
                pred = self.model.predict(np.expand_dims(state, axis=0))
             
                action = np.argmax(pred)
             
                return action

    def play_and_remember(self, episode):
        # This function is used with memory replay to play a round of 
        # space invaders and save all state, action, reward, newstate and done tuples
        # to be sampled randomly during training later
        done = False
        ep_reward = 0
        state = self.env.reset()
        while not done:
            action = self.getAction(state)
            new_state, reward, done, info = self.env.step(action)
            ep_reward += reward
            self.memory.append({"state":state, "action":action, "reward":reward, "new_state":new_state, "done": done})
            state = new_state
        print(f"Episode {episode + 1} results: Reward after terminal =  {ep_reward}, epsilon: {self.epsilon}")


    def train(self, n_episodes=5000):
            # Training function
            if self.replay: 
                # If replay is true, then we play and remember an episode
                for i in range(n_episodes):
                    
                    self.play_and_remember(i)
                    
                    if len(self.memory) > self.batch_size: 
                        # if the memory is sufficiently large, we sample it and
                        # train a batch
                        samples = random.sample(self.memory, self.batch_size)
                        X = []
                        ys = []
                        for sample in samples:
                            _action, _state, _new_state, _reward, _done = sample["action"], sample["state"], sample["new_state"], sample["reward"], sample["done"]
                     
                            if _done:
                                y = _reward
                            else:
                                y = _reward + self.discount * self.getActionValue(_new_state)
                            
                            targ_vec = self.model.predict(np.expand_dims(_state, axis=0))
                            targ_vec[0][_action] = y
                            X.append(_state)
                            ys.append(targ_vec[0])
                          
                        X = np.array(X)
                        ys = np.array(ys)
                        self.model.fit(X, ys, batch_size=self.batch_size, epochs=1, verbose=0)
                    
                    if self.epsilon > self.epsilon_limit:
                        self.epsilon *= self.decay

            else:
                # this is the training loop for non memory replay
                for i in range(n_episodes):
                    
                    state = np.asarray(env.reset())
                    done = False
                    ep_reward = 0
                    while not done:
                     
                        action = self.getAction(state)
                        new_state, reward, done, info = self.env.step(action)
                        new_state = np.asarray(new_state)
                        ep_reward += reward
                        if done:
                            print(f"Episode {i + 1}, Reward after terminal: {ep_reward}, epsilon = {self.epsilon}")
                        else:
                            y = reward + self.discount * self.getActionValue(new_state)
                        
                        targ_vec = self.model.predict(np.expand_dims(state, axis=0))
                        targ_vec[0][action] = y
                        self.model.fit(np.expand_dims(state, axis=0), targ_vec, batch_size=self.batch_size, epochs=1, verbose=0)
                        state = new_state
                    self.epsilon *= self.decay

class Qnet():
    # The Qnet is a convolutional neural net that has a light and not light option
    # this just means there is a choice for a larger or smaller model option
    def __init__(self,in_dim, actions, lr=1e-3, light=False):
        if light:
            self.model = self.build_model_light(in_dim, actions, lr)
        else:
            self.model = self.build_model(in_dim, actions, lr)
    
    def build_model(self, in_dim, actions, lr):
        model = Sequential()
        model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=( 210, 160, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Convolution2D(32, (3, 3), activation='relu'))

        model.add(Flatten())
    
        model.add(Dense(512, activation='relu'))
        model.add(Dense(actions, activation="linear"))
        optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr)
        model.compile(optimizer, loss="mse")
        model.summary()
        return model

    def build_model_light(self, in_dim, actions, lr):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=( 210, 160, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Convolution2D(32, (3, 3), activation='relu'))

        model.add(Flatten())
    
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions, activation="linear"))
        optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr)
        model.compile(optimizer, loss="mse")
        model.summary()
        return model
    