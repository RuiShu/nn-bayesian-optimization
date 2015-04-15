all:
	python -m utilities.optimizer
par:
	mpiexec -np 8 python -m mpi.mpi_optimizer
clean:
	find -name "*pyc" -delete
