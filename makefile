all:
	python -m utilities.optimizer
par:
	mpiexec -np 4 python -m mpi.mpi_optimizer
clean:
	find -name "*pyc" -delete
push:
	git checkout master
	git pull origin master
	git merge development
	git push origin master
	git checkout development
