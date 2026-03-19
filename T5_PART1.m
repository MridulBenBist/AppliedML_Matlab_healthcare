%% BME 516/616 - Assignment #5: Part 1 Finish
% Filename: ANN_PART_1_FINISH.m
clear; clc; close all;
rng(1); % Requirement for reproducibility

%% 1. Dataset Loading & Table Extraction
% Load the preprocessed diabetes dataset 
S = load('Preprocessed_Diabetes_Dataset.mat'); 
varName = fieldnames(S);
data_table = S.(varName{1}); 

% Extract numeric data using curly braces to avoid table-conversion errors
X = data_table{:, 1:8}; % 8 input features 
Y = categorical(data_table{:, 9}); % Outcome variable 

% Data splitting: 70% Train, 15% Val, 15% Test using dividerand 
[trainInd, valInd, testInd] = dividerand(size(X,1), 0.70, 0.15, 0.15);
X_train = X(trainInd, :); Y_train = Y(trainInd, :);
X_val   = X(valInd, :);   Y_val   = Y(valInd, :);
X_test  = X(testInd, :);  Y_test  = Y(testInd, :);

%% 2. Initial ANN Architecture (ANN_1 & ANN_2)
% Architecture defined for items 1a and 1d 
ANN_layers = [
    featureInputLayer(8, 'Name', 'input')
    fullyConnectedLayer(10, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
];

% --- ANN_1: Initial Training (Underfitting) 
% Using 1e-5 so that 1e-3 (for ANN_2) counts as an "increase" 
opts1 = trainingOptions('adam', 'MaxEpochs', 100, 'InitialLearnRate', 1e-5, ...
    'ValidationData', {X_val, Y_val}, 'Plots', 'training-progress', 'Verbose', false);
fprintf('Training ANN_1 (Low LR to show underfitting)...\n');
ANN_1 = trainnet(X_train, Y_train, ANN_layers, 'crossentropy', opts1); 

% --- ANN_2: Increased Learning Rate ---
% Set to 1e-3 as specifically required to mitigate underfitting 
opts2 = opts1;
opts2.InitialLearnRate = 1e-3; 
fprintf('Training ANN_2 (Increased LR to 1e-3)...\n');
ANN_2 = trainnet(X_train, Y_train, ANN_layers, 'crossentropy', opts2); 

%% 3. Optimized ANN (ANN_3)
% Use fitcnet for hyperparameter optimization 
trainTable = array2table(X_train);
trainTable.Outcome = Y_train;
fprintf('Optimizing ANN_3 (Max 100 iterations)... [cite: 76]\n');
ANN_3 = fitcnet(trainTable, 'Outcome', 'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations', 100)); 

%% 4. Extra Credit Improvements 
EC_layers = [
    featureInputLayer(8, 'Name', 'input')
    fullyConnectedLayer(20, 'Name', 'fc1_ec')
    batchNormalizationLayer('Name', 'bn1') 
    reluLayer('Name', 'relu1_ec')
    dropoutLayer(0.2, 'Name', 'drop1')     
    fullyConnectedLayer(2, 'Name', 'fc2_ec')
    softmaxLayer('Name', 'softmax_ec')
];

opts_EC = trainingOptions('adam', 'MaxEpochs', 300, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationPatience', 15, ... % Early Stopping 
    'InitialLearnRate', 1e-3, 'Verbose', false);
fprintf('Training Extra Credit Model...\n');
ANN_EC = trainnet(X_train, Y_train, EC_layers, 'crossentropy', opts_EC);

%% 5. Performance Metrics & Visualization
models = {ANN_1, ANN_2, ANN_3, ANN_EC};
mNames = {'ANN_1', 'ANN_2', 'ANN_3', 'ANN_EC'};

figure('Name', 'ROC Curves - Part 1', 'Color', 'w');
for i = 1:4
    if i == 3 % fitcnet object 
        [Y_pred, scores] = predict(models{i}, X_test);
        Y_pred = categorical(Y_pred);
    else      % trainnet objects 
        scores = predict(models{i}, X_test);
        [~, labels] = max(scores, [], 2);
        Y_pred = categorical(labels-1);
    end
    
    % Metrics: Accuracy, Precision, Recall, F1 
    cm = confusionmat(Y_test, Y_pred);
    acc = sum(diag(cm)) / sum(cm(:));
    prec = cm(2,2) / (cm(2,2) + cm(1,2));
    rec = cm(2,2) / (cm(2,2) + cm(2,1));
    f1 = 2 * (prec * rec) / (prec + rec);
    
    fprintf('\n%s: Acc: %.2f | Prec: %.2f | Rec: %.2f | F1: %.2f\n', mNames{i}, acc, prec, rec, f1);
    
    % Plot ROC Curve 
    subplot(2,2,i);
    [rocX, rocY, ~, aucVal] = perfcurve(Y_test, scores(:,2), '1');
    plot(rocX, rocY, 'LineWidth', 2);
    title(sprintf('%s (AUC: %.2f)', mNames{i}, aucVal));
    grid on; xlabel('FPR'); ylabel('TPR');
end

%% 6. Neuron Activation (Item 1d Fix for R2024b)
% Extract one observation for analysis 
test_vector = X_test(1, :); 

% R2024b Fix: Use predict with 'Outputs' to extract internal activations 
% This replaces the 'activations' function which is incompatible with trainnet objects
acts_fc1 = predict(ANN_2, test_vector, 'Outputs', 'fc1'); 

fprintf('\n--- Item 1d Result ---\n');
fprintf('Activation achieved for Neuron #1 in layer fc1: %.4f\n', acts_fc1(1)); 

% Use analyzeNetwork for Item 1a/2a structural details 
analyzeNetwork(ANN_2);