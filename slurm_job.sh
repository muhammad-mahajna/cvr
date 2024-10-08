#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10
#SBATCH --mem=1GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --partition=cpu16
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out

sleep 10s
echo Hello World
