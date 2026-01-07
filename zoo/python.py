from canonical_toolkit.utils.initialize import *

import canonical_toolkit as ctk

robot = initialize_random_graph()

print(robot)

node = ctk.node_from_graph(robot)

print(node)
