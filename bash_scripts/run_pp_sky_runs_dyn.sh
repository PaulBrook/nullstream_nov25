#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=pp_sky_runs
#SBATCH --account=vecchioa-gw-pta
#SBATCH --qos=bbdefault
#SBATCH --time=0-01:00:00
##SBATCH --array=1
#SBATCH --open-mode=truncate
#SBATCH --output=/rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/slurm_output/sky_maps_realres_noinj_resshuff.o

module purge
module load bluebear
module load Miniconda3/4.9.2

# Activate your virtual environment
source /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/venv_orig/bin/activate

python /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/pp_sky_runs_dyn.py -s /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/sim_config_template.yaml -r /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/run_config_template.yaml -d /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/pta_cake_output/250425_simresidsregtoas/ -o /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/sky_maps_output/250425_simresidsregtoas -p 
