function gp_v_nn()

gp100 = load('gp_100obs_time_data.csv');
gp250 = load('gp_250obs_time_data.csv');
gp300 = load('gp_300obs_time_data.csv');
gp350 = load('gp_350obs_time_data.csv');
gp400 = load('gp_400obs_time_data.csv');
gp450 = load('gp_450obs_time_data.csv');
gp500 = load('gp_500obs_time_data.csv');
nn100 = load('nn_100obs_time_data.csv');
nn250 = load('nn_250obs_time_data.csv');
nn300 = load('nn_300obs_time_data.csv');
nn350 = load('nn_350obs_time_data.csv');
nn400 = load('nn_400obs_time_data.csv');
nn450 = load('nn_450obs_time_data.csv');
nn500 = load('nn_500obs_time_data.csv');

size(gp250)

gp = [gp250' gp300' gp350' gp400' gp450' gp500'];
nn = [nn250' nn300' nn350' nn400' nn450' nn500'];
x = [250 300 350 400 450 500];
clf;
hold on;
plot(x, mean(gp), 'g')
plot(x, mean(nn), 'r')
legend('gp', 'nn')
xlabel('iteration')
ylabel('seconds per interation')
