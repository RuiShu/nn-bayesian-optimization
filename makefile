all:
	python optimizer.py
par:
	mpiexec -np 4 python -m mpi.mpi_optimizer
clean:
	rm *pyc
