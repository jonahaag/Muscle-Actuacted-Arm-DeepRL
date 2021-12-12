clear all;
%% Initialization
% Load muscle parameters
init_extensor_muscle;
init_flexor_muscle;
l_1 = 0.1675*2;
l_2 = 0.1315*2;
x_box = 0.8;
y_box = -0.8;

% Simulation parameters
tsim = 0.75;
release_time = 0.15;
dt = 0.001;
t_span = (0:dt:tsim)';

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

rng(0)

%% Create DDPG agent https://de.mathworks.com/help/reinforcement-learning/ug/create-simulink-environment-and-train-agent.html
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

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-02,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','actorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',dt,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.9, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 10;
agentOpts.NoiseOptions.VarianceDecayRate = 0;
agent = rlDDPGAgent(actor,critic,agentOpts);

maxepisodes = 500;
maxsteps = ceil(tsim/dt);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',10, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',-1200);


trainingStats = train(agent,env,trainOpts);


% save("currentAgent.mat",'agent')
