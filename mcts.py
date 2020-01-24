class node:
    def __init__(self, parent, action):
        self.__action = action
        self.__parent = parent
        self.__wins = 0
        self.__simulations = 0
        self.__children = []

    def add_child(self, action):
        ch = node(self, action)
        self.__children.append(ch)
        return ch

    def is_root(self):
        return not self.__parent

    def is_leaf(self):
        return self.__children == []

    def parent(self):
        return self.__parent

    def action(self):
        return self.__action
