clear all;


load('current_agent.mat')
mdl = 'throwing_arm';

% obsInfo = rlNumericSpec([6 1]);
% obsInfo.Name = 'arm and ball position';
% numObservations = obsInfo.Dimension(1);
% 
% actInfo = rlNumericSpec([3 1],...
%     'LowerLimit',[0 0 -1]',...
%     'UpperLimit',[1 1 1]');
% actInfo.Name = 'actions';
% actInfo.Description = 'input flexor, extensor and release';
% numActions = actInfo.Dimension(1);


env = rlSimulinkEnv(mdl,[mdl '/Agent'],obsInfo,actInfo);
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);