#!/bin/bash

parameters=('scRNA' 'scRNA5p' 'snRNA')


# Loop over each parameter set
for param in "${parameters[@]}"; do
  # Create a unique SLURM script for each parameter set
  script_name="scripts/slurm_script_${param}.sh"
  
  cat <<EOT > $script_name
#!/bin/bash
#SBATCH --job-name=job_$param
#SBATCH --output=log/output_$param.out
#SBATCH --error=log/error_$param.err
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=hpc

module load Anaconda3/2022.10

source activate scanpy

/mnt/beegfs/macera/.conda/envs/scanpy/bin/python /mnt/beegfs/macera/CZI/Downstream/HORIZONTAL_Integration/RNA_final/DEG/DEG.py $param
EOT

  # Submit the generated SLURM script
  sbatch $script_name
done