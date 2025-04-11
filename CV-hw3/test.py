class Agent(object):
    def __init__(self, measure):
        super(Agent, self).__init__()
        self.attr = {}
        for k, v in measure.items():
            if isinstance(v, dict):
                self.attr[k] = Agent(v)
            else:
                self.attr[k] = v

    def __getattr__(self, item):
        return self.attr[str(item)]

    def get_transform(self):
        if 'transform' in self.attr:
            return self.attr['transform']
        else:
            raise Exception

    def __str__(self):
        return self.attr.__str__()

data = {
    'name': 'Agent1',
    'settings': {
        'color': 'blue',
        'transform': [1, 2, 3]
    }
}


def display_agent_info(agent):
    print(f"Agent Name: {agent.name}")
    print(f"Agent Color: {agent.settings.color}")
    print(f"Agent Transform: {agent.settings.get_transform()}")

agent = Agent(data)
print(agent.name)                  # Outputs: Agent1
print(agent.settings.color)        # Outputs: blue
print(agent.settings.get_transform())  # Outputs: [1, 2, 3]