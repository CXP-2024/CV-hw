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

from scipy.spatial.transform import Rotation
import numpy as np
yaw = 0
pitch = 30
roll = 0
nonplayer_rot = Rotation.from_euler(
			'zyx', [yaw, 
					pitch,
					roll], degrees=True)
rotation_matrix = nonplayer_rot.as_matrix()
print(rotation_matrix)  # Outputs the rotation matrix

data = {
                "boundingBox": {
                    "extent": {
                        "x": 1.9876738786697388,
                        "y": 0.8463225364685059,
                        "z": 0.8048363924026489
                    },
                    "transform": {
                        "location": {
                            "z": 0.7699999809265137
                        },
                        "orientation": {
                            "x": 1.0
                        },
                        "rotation": {}
                    }
                },
                "forwardSpeed": 5.594553470611572,
                "transform": {
                    "location": {
                        "x": 1.5918151140213013,
                        "y": 32.22174072265625,
                        "z": 38.148128509521484
                    },
                    "orientation": {
                        "x": -0.0005538463592529297,
                        "y": -0.9999951720237732,
                        "z": -0.0031015125568956137
                    },
                    "rotation": {
                        "pitch": -0.17770785093307495,
                        "roll": 0.0008003832772374153,
                        "yaw": -90.03173065185547
                    }
                }
            }

agent = Agent(data)
print(agent.boundingBox.extent.x)  # Accessing nested attributes
print(agent.boundingBox.transform.location.z)  # Accessing nested attributes
print(agent.get_transform().location.x)  # Accessing nested attributes