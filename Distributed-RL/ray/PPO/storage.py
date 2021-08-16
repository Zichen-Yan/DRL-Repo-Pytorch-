import ray

@ray.remote
class SharedStorage(object):

    def __init__(self, agent):
        self.update_counter = 0
        self.interaction_counter = 0
        self.sample_counter = 0

        # create actor based on config details
        self.agent = agent
        self.evaluation_reward = {}

    def get_weights(self):
        return self.agent.get_weights()

    def set_weights(self, weights):
        return self.agent.set_weights(weights)

    def add_update_counter(self):
        self.update_counter += 1

    def get_update_counter(self):
        return self.update_counter

    def add_sample_counter(self):
        self.sample_counter += 1

    def get_sample_counter(self):
        return self.sample_counter

    def reset_sample_counter(self):
        self.sample_counter = 0

    # def set_eval_reward(self, step, update_number, rewards):
    #     self.evaluation_reward[update_number] = rewards

    def add_interactions(self, steps):
        self.interaction_counter += steps
    
    def get_interactions(self):
        return self.interaction_counter

