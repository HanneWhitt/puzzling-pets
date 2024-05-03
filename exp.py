import numpy as np


class Shape(np.ndarray):

    """
    Describes a specific rotation of a piece, or the board.         
    """

    def __new__(cls, representation):  
        return np.asarray(representation).view(cls)


    def __init__(self, representation):
        #super().__init__()
        self.I = representation.shape[1]
        self.J = representation.shape[2]
        self.length = max(self.I, self.J)


a = np.array([[[1, 2], [3, 4]]])

s = Shape(a)


print(s.length)
print(s.shape)