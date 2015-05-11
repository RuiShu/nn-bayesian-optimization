function gp()

gp100 = load('gp_100obs_time_data.csv');
gp250 = load('gp_250obs_time_data.csv');
gp300 = load('gp_300obs_time_data.csv');
gp350 = load('gp_350obs_time_data.csv');
gp400 = load('gp_400obs_time_data.csv');
gp450 = load('gp_450obs_time_data.csv');
gp500 = load('gp_500obs_time_data.csv');

size(gp250)

gp = [gp250' gp300' gp350' gp400' gp450' gp500'];

x = [250 300 350 400 450 500];
clf;
hold on;
grid on;

plot(x, mean(gp), 'r', 'linewidth', 1)
plot(x, +2*std(gp)+mean(gp), 'r--', 'linewidth', 1)
plot(x, -2*std(gp)+mean(gp), 'r--', 'linewidth', 1)
xlabel('Number of Queries', 'interpreter', 'latex', 'fontsize', 15)
ylabel('Seconds per Iteration', 'interpreter', 'latex', 'fontsize', 15)
h = legend('Gaussian Process', 'location', 'northwest')
set(h, 'interpreter', 'latex', 'fontsize', 15)
