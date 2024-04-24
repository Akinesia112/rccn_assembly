import pathlib
from compas_assembly.algorithms import assembly_interfaces_numpy
from compas_assembly.datastructures import Assembly
from compas_rbe.equilibrium import compute_interface_forces_cvx


CWD = pathlib.Path(__file__).parent.absolute()
# make 'output' in parent directory if it doesn't exist
OUTPUT_DIR = CWD.parent / 'output'
INPUT_FILE_PATH = OUTPUT_DIR / 'assembly_from_rhino_test3.json'
OUTPUT_FILE_PATH = OUTPUT_DIR / 'assembly_interface_from_rhino_test3.json'

assembly = Assembly.from_json(INPUT_FILE_PATH)

assembly_interfaces_numpy(assembly)
compute_interface_forces_cvx(assembly, solver='CPLEX', verbose=True)

assembly.to_json(OUTPUT_FILE_PATH, pretty=True)