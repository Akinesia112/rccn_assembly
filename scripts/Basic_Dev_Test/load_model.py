import compas
from compas.robots import LocalPackageMeshLoader
from compas.robots import RobotModel

import compas_fab

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

# Locate the URDF file inside compas fab installation
path = r"D:\RCCN_Assembly\rccn_assembly\hiwin_ra620_1621_support\urdf\ra620_1621.urdf"
urdf = compas_fab.get(path)

# Create robot model from URDF
model = RobotModel.from_urdf_file(urdf)

# Also load geometry
loader = LocalPackageMeshLoader(compas_fab.get('universal_robot'), 'ur_description')
model.load_geometry(loader)

print(model)