import compas_fab
from compas.geometry import Frame
from compas_fab.backends import PyBulletClient

with PyBulletClient() as client:
    urdf_filename = compas_fab.get(r"D:\RCCN_Assembly\rccn_assembly\hiwin_ra620_1621_support\urdf\ra620_1621.urdf")
    robot = client.load_robot(urdf_filename)

    frame_WCF = Frame([0.3, 0.1, 0.5], [1, 0, 0], [0, 1, 0])
    start_configuration = robot.zero_configuration()
    configuration = robot.inverse_kinematics(frame_WCF, start_configuration)

    print("Found configuration", configuration)