../neon/.venv/bin/python -i train.py GamesList.txt -z 128 --eval 1 -e 150 Workspace/ -b gpu -v -l training.log --serialize 1 -o traindata.hdf5 -H 5 -s Workspace/snapshot.pkl
