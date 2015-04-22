function plotme()

seq = load('sequential_time_data.csv');
mpi = load('mpi_time_data.csv');

seq_y = mean(seq, 1);
mpi_y = mean(mpi, 1);
x = 1:250;

clf;
hold on;
plot(x, seq_y, 'b');
plot(x, mpi_y, 'r');
legend('sequential', 'parallel')
xlabel('iterations')
ylabel('time')
