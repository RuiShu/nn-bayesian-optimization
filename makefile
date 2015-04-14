all:
	python optimizer.py
par:
	mpiexec -np 4 python mpi_optimizer.py
clean:
	rm *pyc
