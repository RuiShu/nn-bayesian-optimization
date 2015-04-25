all:
	python -m utilities.optimizer
seq:
	python -m sequential.seq_optimizer
gp:
	python -m sequential.seq_gaussian_process
par:
	mpiexec -np 8 python -m mpi.mpi_optimizer
clean:
	find -name "*pyc" -delete
push:
	git checkout master
	git pull origin master
	git merge development
	git push origin master
	git checkout development
seqloop:
	for number in 1 2 3 4 5 6 7 8 9 10 ; do \
		python -m sequential.seq_optimizer ; \
	done
parloop:
	for number in 1 2 3 4 5 6 7 8 9 10 ; do \
		mpiexec -np 8 python -m mpi.mpi_optimizer ; \
	done
gploop:
	for number in 1 2 3 4 5 6 7 8 9 10 ; do \
		python -m sequential.seq_gaussian_process ; \
	done
loop:
	for number in 1 2 3 4 5 6 7 8 9 10 ; do \
		python -m sequential.seq_optimizer ; \
	done
	for number in 1 2 3 4 5 6 7 8 9 10 ; do \
		python -m sequential.seq_gaussian_process ; \
	done
