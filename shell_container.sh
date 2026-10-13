export APPTAINER_CMD=/disk/users/`whoami`/apptainer/bin/apptainer
export PROJECTS_DIR="$(dirname "$PWD")"
export SAVEDIR=/scratch/`whoami`
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
apptainer exec --nv -B /cvmfs -B /disk/users/`whoami` -B /disk/gfs_lhcb/`whoami`\
            /disk/users/lprate/containers/MuonsAndMatterContainer.sif bash --rcfile set_env.sh
