import pathlib
from compas_assembly.datastructures import Assembly
from compas.datastructures import Network
from compas_view2.app import App
from compas_view2.objects import Collection
from compas.geometry import Scale

CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "scripts" / "output" / "assembly_interface_from_rhino_hexa.json"

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
viewer.add(Collection(polygons), facecolor=(0,0,0), opacity=0.5)
viewer.view.camera.scale = 2000
viewer.view.camera.position = [3000, 3000, 3000]
viewer.view.camera.distance = 1000
viewer.show()