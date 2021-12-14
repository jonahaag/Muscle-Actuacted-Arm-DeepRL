clear all;
% Load muscle  parameters
init_extensor_muscle;
init_flexor_muscle;
% some parameters
l_1 = 0.1675*2;
l_2 = 0.1315*2;
x_box = 0.6;
y_box = -0.6;
release_time = 0.15;
% simulation parameters
tsim = 1;
dt = 0.001;
t_span = (0:dt:tsim)';
maxsteps = ceil(tsim/dt);
% load agent
load('current_agent.mat')
% load environment
mdl = 'throwing_arm';
obsInfo = rlNumericSpec([9 1]);
obsInfo.Name = 'arm and ball position, velocity and acceleration';
numObservations = obsInfo.Dimension(1);
actInfo = rlNumericSpec(1,...
    'LowerLimit',-1,...
    'UpperLimit',1);
actInfo.Name = 'actions';
actInfo.Description = 'muscle input';
numActions = actInfo.Dimension(1);
env = rlSimulinkEnv(mdl,[mdl '/Agent'],obsInfo,actInfo);
% simulate
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);
