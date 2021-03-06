% Muscle parameters
init_extensor_muscle;
init_flexor_muscle;

% spatial parameters
l_1 = 0.1675*2;
l_2 = 0.1315*2;
x_box = 0.6;
y_box = -0.6;

% predefined release time
release_time = 0.165;
%release_time = 0.5;

%
tsim = 0.8;
dt = 0.001;
t_span = (0:dt:tsim)';
maxsteps = ceil(tsim/dt);