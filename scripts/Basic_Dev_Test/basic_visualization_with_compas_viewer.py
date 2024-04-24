import pathlib
from compas_assembly.datastructures import Assembly
from compas.datastructures import Network
from compas_view2.app import App
from compas_view2.objects import Collection
from compas.geometry import Scale

CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent / "output" / "assembly_interface_from_rhino_1108.json"

assembly = Assembly.from_json(FILE)

polygons = []
for interface in assembly.interfaces():
    polygons.append(interface.polygon)

viewer = App()

for node in assembly.graph.nodes():
    block = assembly.graph.node_attribute(node, 'block')
    viewer.add(block, opacity=0.5)

network = Network.from_networkx(assembly.graph.to_networkx())
print(network.summary())
viewer.add(network)
viewer.add(Collection(polygons), facecolor=(0.3,0,0.3))
viewer.view.camera.scale = 2000
viewer.show()