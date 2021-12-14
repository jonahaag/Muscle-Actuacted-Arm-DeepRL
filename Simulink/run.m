clear all;
%% Initialization
init_throwing_arm

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

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

% figure
% plot(criticNetwork)

actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name', 'ActorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',dt,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.5;
agentOpts.NoiseOptions.VarianceDecayRate = 0;
agent = rlDDPGAgent(actor,critic,agentOpts);

maxepisodes = 500;
maxsteps = ceil(tsim/dt);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',0,...
    'UseParallel',false,...
    'StopOnError','off');
trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "experiences";


trainingStats = train(agent,env,trainOpts);


% save("currentAgent.mat",'agent')
