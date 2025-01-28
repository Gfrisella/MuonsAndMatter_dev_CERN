import json
import numpy as np
import gzip
import pickle
from time import time
from lib.ship_muon_shield_customfield import get_design_from_params, get_field
from muon_slabs import (
    simulate_muon, initialize, collect, 
    kill_secondary_tracks, collect_from_sensitive
)
from plot_magnet import plot_magnet, construct_and_plot

def run(
    muons,
    phi,
    input_dist: float = None,
    return_weight: bool = False,
    fSC_mag: bool = True,
    sensitive_film_params: dict = {'dz': 0.01, 'dx': 10, 'dy': 10, 'position': 67},
    use_field_maps: bool = False,
    field_map_file: str = None,
    return_nan: bool = False,
    seed: int = None,
    draw_magnet: bool = False,
    back_track: bool = False,
    kwargs_plot: dict = {},
):
    # Unpack tuple input if necessary
    if isinstance(muons, tuple):
        muons = muons[0]

    # Initialize detector configuration
    detector = get_design_from_params(
        params=phi,
        force_remove_magnetic_field=False,
        fSC_mag=fSC_mag,
        use_field_maps=use_field_maps,
        sensitive_film_params=sensitive_film_params,
        field_map_file=field_map_file
    )
    detector.update({
        "store_primary": sensitive_film_params is None or back_track,
        "store_all": False
    })

    # Seed initialization
    if seed is None:
        seed_vec = np.random.randint(0, 256, 4).tolist()
    else:
        seed_vec = [seed] * 4

    # Initialize simulation
    t1 = time()
    output_data = json.loads(initialize(*seed_vec, json.dumps(detector)))
    print('Time to initialize:', time() - t1)

    kill_secondary_tracks(False)

    # Unpack muon data
    if muons.shape[-1] == 8:
        px, py, pz, x, y, z, charge, W = muons.T
    else:
        px, py, pz, x, y, z, charge = muons.T
        W = None

    charge = np.sign(charge)  # Normalize charge to +/- 1
    assert (np.abs(charge) == 1).all(), "All charges must be +/- 1."

    # Adjust z-position if needed
    if input_dist is not None:
        z = -input_dist * np.ones_like(z)
    else:
        z_offset = 70.845 - 68.685 + 66.34
        magnet_z = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz']
        z = z / 100 + z_offset + magnet_z

    muon_data = []

    # Simulation loop
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i], y[i], z[i])
        if sensitive_film_params is None:
            muon_data.append(collect())
        else:
            data_s = collect() if back_track else collect_from_sensitive()
            if data_s and valid_muon_hit(data_s, detector, back_track):
                muon_data.append(process_muon_hit(data_s, W[i] if W is not None else None, seed_vec, back_track))
            elif return_nan:
                muon_data.append([0] * muons.shape[-1])

    muon_data = np.asarray(muon_data)

    # Plot magnet if requested
    if draw_magnet:
        plot_magnet(detector, muon_data=muon_data, sensitive_film_position=5, **kwargs_plot)

    return (muon_data, output_data['weight_total']) if return_weight else muon_data

def valid_muon_hit(data_s, detector, back_track):
    if back_track:
        z_pos = detector['sensitive_film']['z_center'] + 15
        return (
            len(data_s['px']) > 0 and 13 in np.abs(data_s['pdg_id']) and
            data_s['z'][-1] >= z_pos and
            np.abs(data_s['x'][-1]) <= detector['sensitive_film']['dx'] and
            np.abs(data_s['y'][-1]) <= detector['sensitive_film']['dy']
        )
    return len(data_s['px']) > 0 and 13 in np.abs(data_s['pdg_id'])

def process_muon_hit(data_s, weight, seed_vec, back_track):
    j = next((i for i, pdg in enumerate(data_s['pdg_id']) if abs(pdg) == 13), None)
    output = [
        data_s['px'][j], data_s['py'][j], data_s['pz'][j],
        data_s['x'][j], data_s['y'][j], data_s['z'][j],
        data_s['pdg_id'][j]
    ]
    if weight is not None:
        output.append(weight)
        data_s['W'] = weight
    data_s.update({"weight_total": None, "seed": seed_vec})
    return data_s if back_track else output

def split_array(arr, num_chunks):
    base_size = len(arr) // num_chunks
    remainder = len(arr) % num_chunks
    sizes = [base_size + 1 if i < remainder else base_size for i in range(num_chunks)]
    return np.split(arr, np.cumsum(sizes)[:-1])

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    from lib.reference_designs.params import sc_v6

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--c", type=int, default=45)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("--f", type=str, default='data/inputs.pkl')
    parser.add_argument("-tag", type=str, default='geant4')
    parser.add_argument("-params", nargs='+', default=sc_v6)
    parser.add_argument("--z", type=float, default=0.9)
    parser.add_argument("-sens_plane", type=float, default=None)
    parser.add_argument("-real_fields", action='store_true')
    parser.add_argument("-field_file", type=str, default='data/outputs/fields.pkl')
    parser.add_argument("-shuffle_input", action='store_true')
    parser.add_argument("-plot_magnet", action='store_true')
    parser.add_argument("-warm", dest="SC_mag", action='store_false')
    parser.add_argument("-return_nan", action='store_true')
    parser.add_argument("-back_track", action='store_true')
    parser.add_argument("-save_output", action='store_true')

    args = parser.parse_args()
    cores = args.c
    params = list(args.params)

    # Load data
    with gzip.open(args.f, 'rb') as f:
        data = pickle.load(f)

    if args.shuffle_input:
        np.random.shuffle(data)

    if 0 < args.n <= len(data):
        data = data[:args.n]
        cores = min(cores, len(data))

    workloads = split_array(data, cores)

    # Run simulation
    t1 = time()
    with mp.Pool(cores) as pool:
        results = pool.starmap(
            run, [
                (workload, params, args.z, True, args.SC_mag, None if args.sens_plane is None else {
                    'dz': 0.01, 'dx': 4, 'dy': 6, 'position': args.sens_plane
                }, args.real_fields, args.field_file, args.return_nan, args.seed, False, args.back_track)
                for workload in workloads
            ]
        )
    print(f"Simulation completed in {time() - t1:.2f} seconds.")

    # Combine results
    all_results = [res[0] for res in results if len(res[0]) > 0]
    if all_results:
        all_results = np.concatenate(all_results)
        print('Final data shape:', all_results.shape)

        if args.save_output:
            output_path = f"data/outputs/outputs_{args.tag}_{np.random.randint(256)}.pkl"
            with gzip.open(output_path, "wb") as f:
                pickle.dump(all_results, f)

    if args.plot_magnet:
        construct_and_plot(all_results, params, True, {'position': 5}, False, None)
