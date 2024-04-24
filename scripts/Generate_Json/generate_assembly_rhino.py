import os

from compas.datastructures import mesh_weld
from compas_assembly.datastructures import Assembly
from compas_assembly.datastructures import Block
from compas_rhino.utilities import get_meshes

SUPPORT_LAYER_NAME = "Supports"
BLOCK_LAYER_NAME = "Blocks"
OUTPUT_DIR_NAME = "output"
OUTPUT_FILE_NAME = "assembly_from_rhino.json"

# use os.path since pathlib is not available in Rhino IronPython
SCRIPT_FOLDER = os.path.dirname(__file__)
OUTPUT_DIR_PATH = os.path.join(SCRIPT_FOLDER, "..", OUTPUT_DIR_NAME)
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, OUTPUT_FILE_NAME)

# mkdir if output is not present
if not os.path.exists(OUTPUT_DIR_PATH):
    os.mkdir(OUTPUT_DIR_PATH)

# get support meshes and block meshes from Rhino
rhino_supports = get_meshes(SUPPORT_LAYER_NAME)
rhino_blocks = get_meshes(BLOCK_LAYER_NAME)

#  create and weld compas meshes
supports = []
support_centroids = []
for i in rhino_supports:
    support = Block.from_rhinomesh(i)
    support_centroids.append(support.centroid())
    # support = mesh_weld(support)
    supports.append(support)

blocks = []
block_centroids = []
for i in rhino_blocks:
    block = Block.from_rhinomesh(i)
    block_centroids.append(block.centroid())
    # block = mesh_weld(block)
    blocks.append(block)

# create an assembly
assembly = Assembly()

for i, block in enumerate(supports):
    assembly.add_block(block,
                       x=support_centroids[i].x,
                       y=support_centroids[i].y,
                       z=support_centroids[i].z,
                       is_support=True)

for i, block in enumerate(blocks):
    assembly.add_block(block,
                       x=block_centroids[i].x,
                       y=block_centroids[i].y,
                       z=block_centroids[i].z,
                       is_support=False)

assembly.to_json(OUTPUT_FILE_PATH, pretty=True)

# prompts user the path of the assembly file
import rhinoscriptsyntax as rs
rs.Prompt("file is saved to: " + OUTPUT_FILE_PATH)
print(OUTPUT_FILE_PATH)
