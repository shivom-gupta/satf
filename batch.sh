#!/bin/bash
#
#SBATCH --job-name=test_batch
#SBATCH --output=output_batch.txt
#
#SBATCH --ntasks=1
#SBATCH --time=01:30:00

i=20
while [ "$i" -le 23 ]; do
	echo "$i"
	srun --mem=156g --gres=gpu:1g.5gb --container-name="webArchive3" --container-writable --container-mounts=/mnt/ceph/storage/data-tmp/teaching-current/ru07neqa/:/home/ home/bdlt/WARC-DL2/satf/auto.sh "$i"
	i=$(( i + 1 ))
done
