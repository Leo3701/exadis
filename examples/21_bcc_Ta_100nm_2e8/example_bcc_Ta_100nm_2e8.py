import os, sys
import numpy as np

# Import the Python package from either the source example tree or projects/.
script_dir = os.path.dirname(os.path.abspath(__file__))
for relative_path in ('../../python', '../../core/exadis/python'):
    candidate = os.path.abspath(os.path.join(script_dir, relative_path))
    if os.path.isfile(os.path.join(candidate, 'pyexadis_base.py')):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break
else:
    raise ImportError('Cannot locate the pyexadis Python package')
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def example_bcc_Ta_100nm_2e8():
    """example_bcc_Ta_100nm_2e8:
    Example of a 100nm MD-like simulation of bcc Ta loaded
    at a strain rate of 2e8/s.
    E.g. see Bertin et al., Acta Materialia 271, 119884
    """
    
    pyexadis.initialize()

    slip_family = 110  # Select 110, 112, or 123.
    family_selector = {110: 1, 112: 4, 123: 5}
    if slip_family not in family_selector:
        raise ValueError("slip_family must be 110, 112, or 123")
    
    state = {
        "crystal": 'bcc',
        "num_bcc_plane_families": family_selector[slip_family],
        "use_glide_planes": 1,
        "enforce_glide_planes": 1,
        "burgmag": 2.85e-10,
        "mu": 55.0e9,
        "nu": 0.339,
        "a": 1.0,
        "maxseg": 15.0,
        "minseg": 3.0,
        "rtol": 0.3,
        "rann": 0.6,
        "nextdt": 5e-13,
    }
    
    Lbox = 300.0
    G = ExaDisNet()
    G.generate_prismatic_config(state["crystal"], Lbox, 12, 0.21*Lbox,
                                state["maxseg"], uniform=True,
                                plane_family=slip_family)
    net = DisNetManager(G)
    
    vis = None
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=2600.0, Mscrew=20.0, Mclimb=1e-4, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Trapezoid', multi=10, state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint, 
                              collision=collision, topology=topology, remesh=remesh, vis=vis,
                              loading_mode='strain_rate', erate=2e8, edir=np.array([0.,0.,1.]),
                              max_step=10000, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=2, plot_pause_seconds=0.0001,
                              write_freq=100,
                              write_dir=f'output_bcc_Ta_100nm_2e8_{slip_family}')
    sim.run(net, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    example_bcc_Ta_100nm_2e8()
