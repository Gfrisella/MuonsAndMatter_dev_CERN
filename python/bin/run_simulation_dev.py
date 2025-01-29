import json
import numpy as np
import gzip
import pickle
from time import time
from lib.ship_muon_shield_customfield import get_design_from_params, get_field, initialize_geant4
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
    add_cavern = True,
    use_field_maps: bool = False,
    field_map_file: str = None,
    return_nan: bool = False,
    seed: int = None,
    draw_magnet: bool = False,
    back_track: bool = False,
    SmearBeamRadius:float = 5., #cm
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
        field_map_file=field_map_file,
        add_cavern = add_cavern
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
    output_data = initialize_geant4(detector, seed)
    #output_data = json.loads(initialize(*seed_vec, json.dumps(detector)))
    print('Time to initialize:', time() - t1)
    output_data = json.loads(output_data) 
    kill_secondary_tracks(True)

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
    muon_data = []

    # Simulation loop
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i], y[i], z[i], SmearBeamRadius, seed if seed else np.random.randint(0,100))
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
        z_pos = detector['sensitive_film']['z_center']
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
    parser.add_argument("--f", type=str, default='data/enriched_input.pkl')
    parser.add_argument("-tag", type=str, default='geant4')
    parser.add_argument("-params", type=str, default='sc_v6')
    parser.add_argument("--z", type=float, default=None)
    parser.add_argument("-sens_plane", type=float, default=82)
    parser.add_argument("-real_fields", action='store_true')
    parser.add_argument("-field_file", type=str, default='data/outputs/fields.pkl')
    parser.add_argument("-shuffle_input", action='store_true')
    parser.add_argument("-remove_cavern", dest = "add_cavern", action = 'store_false')
    parser.add_argument("-plot_magnet", action='store_true')
    parser.add_argument("-warm", dest="SC_mag", action='store_false')
    parser.add_argument("-return_nan", action='store_true')
    parser.add_argument("-back_track", action='store_true')
    parser.add_argument("-save_output", action='store_true')

    args = parser.parse_args()
    tag = args.tag
    cores = args.c
    
    if args.params == 'sc_v6': params = sc_v6
    elif args.params == 'oliver': params = optimal_oliver
    elif args.params == 'oliver_scaled': params = oliver_scaled
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])
        params_idx = new_parametrization['M1'] + new_parametrization['M2'] + new_parametrization['M3'] + new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']
        params = [
        231.0, 145.88714376, 144.97327917, 233.53443056, 185.12337627, 289.12393279, 178.27166603,
        50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 0.0,
        55.68631679, 39.18737101, 39.0519151, 64.94001798, 2.18559108, 2.24933776, 1.81442009, 0.0,
        41.48397928, 29.13140211, 26.76256635, 115., 2.04579784, 2.23057252, 1.83378473, 0.0,
        8.53668902, 24.10068397, 20.92353585, 18.79892067, 100.0, 2.1116541, 1.85945483, 0.0,
        5.0, 24.87859129, 30.70114222, 15.07063979, 2.25817831, 2.24777563, 1.86769417, 0.0,
        41.48397928, 29.13140211, 26.76256635, 115., 2.04579784, 2.23057252, 1.83378473, 0.0,#17.42397096, 24.81168458, 57.660231, 95.16186569, 2.2578048, 2.0182731, 1.84003594, 0.0,
        41.48397928, 29.13140211, 26.76256635, 115., 2.04579784, 2.23057252, 1.83378473, 0.0]
        #25.59295856, 59.75732946, 47.92959263, 50.68990891, 2.22702035, 2.05949534, 1.80882961, 0.0]
    params = [2.31000000e+02, 1.66443436e+02, 2.56261292e+02, 2.91919128e+02,
                1.99210968e+02, 2.14869919e+02, 1.13462059e+02, 5.00000000e+01,
                5.00000000e+01, 1.19000000e+02, 1.19000000e+02, 2.00000000e+00,
                2.00000000e+00, 1.00000000e+00, 0.00000000e+00, 7.17344208e+01,
                7.23064270e+01, 3.01792870e+01, 5.50608139e+01, 4.50294533e+01,
                6.05307674e+00, 1.55043912e+00, 2.95457952e-02, 7.22414017e+01,
                3.28486481e+01, 6.40336609e+01, 9.86102371e+01, 3.72791176e+01,
                3.75814209e+01, 1.39339948e+00, 0.00000000e+00, 5.00000000e+00,
                3.68521347e+01, 2.36333714e+01, 6.25332870e+01, 4.90338936e+01,
                4.02122650e+01, 1.60776615e+00, 1.52162361e+00, 5.47383785e+00,
                4.36184082e+01, 4.75524864e+01, 1.64372299e+02, 3.82776070e+01,
                2.00000000e+00, 1.26237798e+00, 4.06472445e-01, 2.14549942e+01,
                4.87760811e+01, 5.87977676e+01, 1.13127258e+02, 6.34082699e+00,
                4.75470161e+01, 1.53402686e+00, 0.00000000e+00, 7.60394669e+01,
                2.79700508e+01, 9.10693283e+01, 2.94182072e+01, 2.28603382e+01,
                1.99502850e+01, 9.39842820e-01, 0.00000000e+00]
    params = np.array(params)
    if params.size != 63:
        new_phi = np.array(sc_v6, dtype=params.dtype)
        new_phi[np.array(params_idx)] = params
        if args.SC_mag:
            new_phi[new_parametrization['M2'][2]] = new_phi[new_parametrization['M2'][1]]
            new_phi[new_parametrization['M2'][4]] = new_phi[new_parametrization['M2'][3]]
        params = new_phi
        
    n_muons = args.n
    input_file = args.f
    input_dist = args.z
    if args.sens_plane is not None: 
        sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}
    else: sensitive_film_params = None
    
    
    t1_fem = time()
    detector = None
    if args.real_fields: 
        if os.path.exists(args.field_file):
            os.remove(args.field_file)
        detector = get_design_from_params(np.asarray(params), args.SC_mag, False,True, args.field_file, sensitive_film_params, False, cores_field=cores)
    t2_fem = time()

    # Load data
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    if args.shuffle_input:
        np.random.shuffle(data)

    if 0 < args.n <= len(data):
        data_n = data[:args.n]
        cores = min(cores, len(data))
    else: data_n = data

    workloads = split_array(data_n, cores)
    print(np.shape(data_n))
    print(len(workloads))
    # Print the shape of each element in workloads
    for i, workload in enumerate(workloads):
        print(f"Shape of workload {i}: {np.shape(workload)}")

    # Run simulation
    t1 = time()
    with mp.Pool(cores) as pool:
        results = pool.starmap(
            run, [
                (workload, params, input_dist, True, args.SC_mag, sensitive_film_params, args.add_cavern, args.real_fields, args.field_file, args.return_nan, args.seed, False, args.back_track)
                for workload in workloads
            ]
        )
    print(f"Time to FEM: {t2_fem - t1_fem:.2f} seconds.")
    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {time() - t1:.2f} seconds.")

    # Combine results
    all_results = [res[0] for res in results if len(res[0]) > 0]
    if all_results:
        all_results = np.concatenate(all_results)
        print(all_results)
        print('Final data shape:', all_results.shape)

        if args.save_output:
            output_path = "data/outputs/outputs_warm.pkl"
            with gzip.open(output_path, "wb") as f:
                pickle.dump(all_results, f)

    if args.plot_magnet:
        sensitive_film_params['position'] = 38
        angle = 90
        elev = 0
        if False:#detector is not None:
            plot_magnet(detector, muon_data = all_results, sensitive_film_position = sensitive_film_params['position'], azim = angle, elev = elev)
        else:
            result = construct_and_plot(muons = all_results,phi = params,fSC_mag = args.SC_mag,sensitive_film_params = sensitive_film_params, use_field_maps=args.real_fields, field_map_file = args.field_file, cavern = False, azim = angle, elev = elev)#args.add_cavern)
                                         
