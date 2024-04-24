import pathlib

from compas_assembly.datastructures import Assembly, Block
from compas_assembly.algorithms import assembly_interfaces_numpy
from compas_assembly.geometry import Arch
from compas_rbe.equilibrium import compute_interface_forces_cvx

from datetime import datetime
from compas.datastructures import Mesh
from compas.geometry import transform_points, Transformation
from compas.geometry import Translation, Vector

# 拱門的參數
rise = 1500
span = 3000
thickness = 230
depth = 60  # 原始深度
n = 50
# construct an arch assembly
# 創建第一層拱門
arch1 = Arch(rise=rise, span=span, thickness=thickness, depth=depth, n=n)
assembly1 = Assembly.from_template(arch1)

# 創建第二層拱門，深度與第一層相同
arch2 = Arch(rise=rise, span=span, thickness=thickness, depth=depth, n=n)
assembly2 = Assembly.from_template(arch2)

# 將第二層拱門沿著Y軸移動，使兩層緊密貼合
translation_vector = Vector(0, depth, 0)
transform = Translation.from_vector(translation_vector)

for block in assembly2.blocks():
    block.transform(transform)
    assembly1.add_block(block)

# define the boundary conditions
assembly1.graph.node_attribute(0, "is_support", True)
assembly1.graph.node_attribute(49, "is_support", True)
assembly_interfaces_numpy(assembly1, nmax=10, amin=0.0001) #nmax is the maximum number of iterations, amin is the minimum area of the interface


# compute the equilibrium geometry
compute_interface_forces_cvx(assembly1, solver='CPLEX', verbose=True)

# export the assembly to json
# get current working directory
cwd = pathlib.Path(__file__).parent.absolute()
# make 'output' in parent directory if it doesn't exist
output_dir = cwd.parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

# compose filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filepath = output_dir / 'equilibrium_arch_{}.json'.format(timestamp)
assembly1.to_json(filepath, pretty=True)

# Visualization
from compas_view2.app import App
from compas_view2.objects import Object, NetworkObject

# Object.register(Block, MeshObject)
Object.register(assembly1.graph, NetworkObject)

viewer = App()
# viewer.add(assembly.graph)
for node in assembly1.graph.nodes():
    block = assembly1.graph.node_attribute(node, 'block')
    viewer.add(block)
viewer.show()

def write_meshes_to_obj(meshes, filename):
    with open(filename, 'w') as file:
        file.write("# OBJ file\n")
        vert_offset = 1  # OBJ files are 1-indexed
        for i, mesh in enumerate(meshes):
            file.write(f"o Mesh_{i}\n")  # Object name
            for vertex in mesh.vertices():
                x, y, z = mesh.vertex_coordinates(vertex)
                file.write(f"v {x} {y} {z}\n")
            for face in mesh.faces():
                face_vertices = [str(vertex + vert_offset) for vertex in mesh.face_vertices(face)]
                file.write("f " + " ".join(face_vertices) + "\n") 
            vert_offset += mesh.number_of_vertices()

# Assuming 'assembly' is your Assembly instance
meshes = []
for block in assembly1.blocks():
    block_vertices = [block.vertex_coordinates(vertex) for vertex in block.vertices()]
    block_faces = [block.face_vertices(face) for face in block.faces()]
    meshes.append(Mesh.from_vertices_and_faces(block_vertices, block_faces))

# Export to OBJ file
filepath = r"C:\Users\Acer\Downloads\combined_mesh.obj"
write_meshes_to_obj(meshes, filepath)
