# export PATH="/home/$USER/usr/bin:$PATH"
# export LD_LIBRARY_PATH="/home/$USER/usr/lib:$LD_LIBRARY_PATH"

import g2o_file_reader
from load_2d_g2o import load_2d_g2o
import os


g2o_file = os.path.abspath("./data/slam/input_INTEL_g2o.g2o")
nodes, edges = load_2d_g2o(filename=g2o_file)
print(nodes[1]["id"])
print(nodes[1]["state"])
