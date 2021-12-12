clear all;
init_extensor_muscle;
init_flexor_muscle;
l_1 = 0.1675*2;
l_2 = 0.1315*2;
x_box = 0.8;
y_box = -0.8;

% Simulation parameters
tsim = 2;
release_time = 0.15;
dt = 0.001;
t_span = (0:dt:tsim)';
maxsteps = tsim/dt;

load('currentAgent.mat')
mdl = 'throwing_arm';

obsInfo = rlNumericSpec([6 1]);
obsInfo.Name = 'arm and ball position';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([3 1],...
    'LowerLimit',[0 0 -1]',...
    'UpperLimit',[1 1 1]');
actInfo.Name = 'actions';
actInfo.Description = 'input flexor, extensor and release';
numActions = actInfo.Dimension(1);


env = rlSimulinkEnv(mdl,[mdl '/RL Agent'],obsInfo,actInfo);
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);