#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=pta_cake
#SBATCH --account=vecchioa-gw-pta
#SBATCH --qos=bbdefault
#SBATCH --time=07-15:00:00
##SBATCH --array=1
#SBATCH --open-mode=truncate
##SBATCH --output=/rds/projects/v/vecchioa-gw-pta/brookp/clean/pulsar_dropout/nanograv_15yr_new/slurm_output/param_est_bf_fixed_%j.o
#SBATCH --output=/rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/slurm_output/031125_test.o

module purge
module load bluebear
module load Miniconda3/4.9.2

# Activate your virtual environment
source /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/venv_orig/bin/activate

python /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/run_cake_mpta.py -s /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/sim_config_template.yaml -r /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/scripts/run_config_template.yaml -o /rds/projects/v/vecchioa-gw-pta/brookp/clean/NullStreams_orig/pta_cake_output/031125_test
