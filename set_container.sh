#. /disk/lhcb/scripts/lhcb_setup.sh
export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`
export APPTAINER_TMPDIR=/disk/users/`whoami`/temp
export TMPDIR=/disk/users/`whoami`/tmp
export APPTAINER_CMD=/disk/users/gfrise/Project/containers/install-dir/binapptainer
export PROJECTS_DIR=/disk/users/`whoami`/Project/Diluted_dev/
apptainer shell -B /cvmfs -B /disk/users/`whoami`/Project/Diluted_dev /disk/users/lprate/containers/snoopy_geant.sif
