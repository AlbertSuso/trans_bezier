# Run the experiments on a for loop

for i in 0 1 2 3
do
    sbatch slurm.sh $i
    sleep 2
done

