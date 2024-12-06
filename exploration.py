from openmm.app import *
from openmm import *
from openmm import unit as u
import sys
import mdtraj as md
import numpy as np
from tqdm import tqdm

# Nombre del archivo PDB del péptido
pdb_filename = "pdbs/V5.pdb"
run_index = 0
steps_segment = 1000
n_segments = 1000
temperature = 500 * u.kelvin

# Cargar la estructura del péptido desde el archivo PDB
pdb = PDBFile(pdb_filename)

# Definir la fuerza de campo (AMBER ff14SB es común para péptidos)
forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")

# Crear el sistema en vacío (sin agua ni iones)
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=NoCutoff,  # Sin tratamiento de fronteras periódicas
    constraints=HBonds         # Restringe enlaces de hidrógeno para un paso de integración más largo
)

# Configurar el integrador (Langevin dynamics)
friction = 1 / u.picosecond
timestep = 2 * u.femtoseconds
integrator = LangevinIntegrator(temperature, friction, timestep)

# Crear una plataforma para ejecutar la simulación (e.g., CUDA si tienes GPU)
platform = Platform.getPlatformByName("CUDA")  # Usa "CPU" si no tienes GPU

# Configurar la simulación
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy()

# Minimización de energía
simulation.minimizeEnergy()

md_topology = md.Topology.from_openmm(simulation.topology)

def es_el_mismo(angs1, angs2):
    if np.max(np.abs(angs1-angs2))<0.1:
        return True
    else:
        return False

traj_inh = []
dihed_db = []
coors_db = []
energy_db = []

for ii in tqdm(range(n_segments)):
    simulation.step(steps_segment)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    simulation.minimizeEnergy()
    min_state = simulation.context.getState(getEnergy=True, getPositions=True)
    min_energy = min_state.getPotentialEnergy()
    min_positions = min_state.getPositions(asNumpy=True)
    traj = md.Trajectory(min_positions / u.nanometer, md_topology)
    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    dihed_angs=np.concatenate((phis[0],psis[0]))
    visitado = False
    for unique_index in range(len(dihed_db)):
        aux = es_el_mismo(dihed_angs, dihed_db[unique_index])
        if aux == True:
            visitado = True
            traj_inh.append(unique_index)
            break
    if visitado == False:
        traj_inh.append(len(dihed_db))
        dihed_db.append(dihed_angs)
        coors_db.append(min_positions._value)
        energy_db.append(min_energy._value)

import pickle as pickle
import gzip

with gzip.open(f'energies_{run_index}.pkl.gz', 'wb') as fff:
    pickle.dump(energy_db, fff)

with gzip.open(f'traj_inh_{run_index}.pkl.gz', 'wb') as fff:
    pickle.dump(traj_inh, fff)

with gzip.open(f'coors_{run_index}.pkl.gz', 'wb') as fff:
    pickle.dump(coors_db, fff)

with gzip.open(f'dihed_{run_index}.pkl.gz', 'wb') as fff:
    pickle.dump(dihed_db, fff)

