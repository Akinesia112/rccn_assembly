import compas
from compas.robots import GithubPackageMeshLoader, DefaultMeshLoader, LocalPackageMeshLoader
from compas.robots import RobotModel
from compas_view2.app import App
from compas_view2.collections import Collection
import compas
from compas.robots import LocalPackageMeshLoader
from compas.robots import RobotModel
import itertools
from compas.geometry import Transformation
from compas.geometry import Translation
from compas.geometry import Rotation
from compas.robots import Geometry
from math import radians, degrees
import compas_fab

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

# model = RobotModel.ur5(True)

#github = GithubPackageMeshLoader('ros-industrial/abb', 'abb_irb6600_support', 'kinetic-devel')
#model = RobotModel.from_urdf_file(github.load_urdf('irb6640.urdf'))
#model.load_geometry(github)

#github = GithubPackageMeshLoader('rccn-dev/kuka_experimental', 'kuka_kr300_support', 'melodic-devel')
#model = RobotModel.from_urdf_file(github.load_urdf('kr300r2500pro.urdf'))
#model.load_geometry(github)


#github = GithubPackageMeshLoader('Akinesia112/rccn_assembly', 'hiwin_ra620_1621_support', 'main')
#model = RobotModel.from_urdf_file(github.load_urdf('ra620_1621.urdf'))
#model.load_geometry(github)

# Locate the URDF file inside compas fab installation
#path = r"D:\RCCN_Assembly\rccn_assembly\hiwin_ra620_1621_support\urdf\ra620_1621.urdf"
path = r"C:\Users\Acer\Downloads\urdf_model-main\urdf_model-main\hiwin_robot_arm\combine_description\hiwin_arm\Hiwin_RA610_1476_GC.urdf"
description_path = r"C:\Users\Acer\Downloads\urdf_model-main\urdf_model-main\hiwin_robot_arm\combine_description\hiwin_arm\combine_description"
#urdf = compas_fab.get(path)
# Create robot model from URDF
model = RobotModel.from_urdf_file(path)
# Also load geometry
loader = LocalPackageMeshLoader(path, description_path)
model.load_geometry(loader)

viewer = App(viewmode="lighted", enable_sceneform=True, enable_propertyform=True, enable_sidebar=True, width=2000, height=1000)

def create(link, parent=None, parent_joint=None):

    obj = None

    meshes = []

    for item in itertools.chain(link.visual):
        meshes.extend(Geometry._get_item_meshes(item))

    obj = parent.add(Collection(meshes), name=link.name, show_lines=False)

    if parent_joint:
        obj.matrix = Transformation.from_frame(parent_joint.origin).matrix

        @viewer.slider(title=parent_joint.name, minval=-180 , maxval=180)
        def rotate(angle):
            T = Translation.from_vector(obj.translation)
            R = Rotation.from_axis_and_angle(parent_joint.axis.vector, radians(angle))
            obj.matrix = (T * R).matrix
            viewer.view.update()

    for joint in link.joints:
        create(joint.child_link, parent = obj, parent_joint=joint)

create(model.root, viewer)
#viewer.view.camera.scale = 2000
#viewer.view.camera.position = [1000, 1000, 1000]
viewer.show()