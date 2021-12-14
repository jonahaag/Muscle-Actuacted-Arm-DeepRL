clear all;
%% Initialization
% Load initialization parameters
init_throwing_arm
% Specify simulink model
mdl = 'throwing_arm';
% Specify observations
obsInfo = rlNumericSpec([9 1]);
obsInfo.Name = 'arm and ball position, velocity and acceleration';
numObservations = obsInfo.Dimension(1);
% Specify actions
actInfo = rlNumericSpec(1,...
    'LowerLimit',-1,...
    'UpperLimit',1);
actInfo.Name = 'actions';
actInfo.Description = 'muscle input';
numActions = actInfo.Dimension(1);
% Define environment
env = rlSimulinkEnv(mdl,[mdl '/Agent'],obsInfo,actInfo);
% Fix random seed
rng(0)

%% Create DDPG agent
% Define different paths for action and state in the critic network
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(50,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];
% Define the critic network
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% Create critic
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

% Uncomment to plot critic network
% figure
% plot(criticNetwork)

% Define actor network
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name', 'ActorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];
% Create actor
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
% Specify agent options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',dt,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-03;
agent = rlDDPGAgent(actor,critic,agentOpts);
% Specify training options
maxepisodes = 50;
maxsteps = ceil(tsim/dt);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',7, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',0,...
    'UseParallel',false,...
    'StopOnError','off');
trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "experiences";

%% Training
trainingStats = train(agent,env,trainOpts);

% Uncomment to save the agent after training is finished, test with
% test_current_agent.m
%save("current_agent.mat",'agent')
