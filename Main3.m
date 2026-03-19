%% BME 516/616 - Assignment #3: Supervised Machine Learning for Diabetes Prediction
clear; clc; close all;
rng(1);

%% SECTION 1: DATA PREPROCESSING

data = readtable('Patient_Data_Diabetes.csv');

% Handle Missing Values (replace zeros with median)
missingVars = {'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'};
for i = 1:length(missingVars)
    varName = missingVars{i};
    nonZeroValues = data.(varName)(data.(varName) ~= 0);
    data.(varName)(data.(varName) == 0) = median(nonZeroValues);
end

% Separate predictors and response
featureNames = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', ...
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'};
X = data{:, featureNames};
Y = data.Outcome;

% Descriptive Statistics
fprintf('--- Item 1c: Descriptive Statistics ---\n');
fprintf('%-30s %10s %10s %10s %10s\n', 'Variable', 'Mean', 'StdDev', 'Min', 'Max');
fprintf('%s\n', repmat('-', 1, 70));
allVarNames = [featureNames, {'Outcome'}];
allData = [X, Y];
for i = 1:size(allData, 2)
    fprintf('%-30s %10.2f %10.2f %10.2f %10.2f\n', ...
        allVarNames{i}, mean(allData(:,i)), std(allData(:,i)), ...
        min(allData(:,i)), max(allData(:,i)));
end

% Feature Scaling
X_scaled = (X - mean(X)) ./ std(X);

%% SECTION 2: LINEAR SVM

fprintf('\n===== LINEAR SVM =====\n');

mdl_linear = fitcsvm(X_scaled, Y, ...
    'KernelFunction', 'linear', ...
    'OptimizeHyperparameters', {'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', struct('KFold', 5, ...
        'MaxObjectiveEvaluations', 30, 'ShowPlots', false, 'Verbose', 0));

%% SECTION 3: NONLINEAR SVM (TEST MULTIPLE KERNELS)
fprintf('\n===== NONLINEAR SVM =====\n');

kernelTypes = {'rbf', 'polynomial'};
bestError = inf;

for k = 1:length(kernelTypes)
    fprintf('\n--- Testing kernel: %s ---\n', kernelTypes{k});
    temp_mdl = fitcsvm(X_scaled, Y, ...
        'KernelFunction', kernelTypes{k}, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct('KFold', 5, ...
            'MaxObjectiveEvaluations', 30, 'ShowPlots', false, 'Verbose', 0));

    temp_loss = kfoldLoss(crossval(temp_mdl, 'KFold', 5));

    if temp_loss < bestError
        bestError = temp_loss;
        bestKernel = kernelTypes{k};
        mdl_nonlinear = temp_mdl;
    end
end

%% SECTION 4: DECISION TREE 1 - OPTIMIZE MaxNumSplits

fprintf('\n===== DECISION TREE 1: Tuning MaxNumSplits =====\n');

mdl_tree_splits = fitctree(X_scaled, Y, ...
    'OptimizeHyperparameters', {'MaxNumSplits'}, ...
    'HyperparameterOptimizationOptions', struct('KFold', 5, ...
        'MaxObjectiveEvaluations', 30, 'ShowPlots', false, 'Verbose', 0));

%% SECTION 5: DECISION TREE 2 - OPTIMIZE MinLeafSize
fprintf('\n===== DECISION TREE 2: Tuning MinLeafSize =====\n');

mdl_tree_leaf = fitctree(X_scaled, Y, ...
    'OptimizeHyperparameters', {'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', struct('KFold', 5, ...
        'MaxObjectiveEvaluations', 30, 'ShowPlots', false, 'Verbose', 0));

%% SECTION 6: NAIVE BAYES (NO HYPERPARAMETER TUNING)
fprintf('\n===== NAIVE BAYES =====\n');

mdl_nb = fitcnb(X_scaled, Y);
fprintf('Naive Bayes model trained (no hyperparameter tuning required).\n');

%% ECTION 7: HYPERPARAMETER RESULTS (Item 2b)

fprintf('\n--- Item 2b: Hyperparameter Tuning Results ---\n\n');

fprintf('Linear SVM:\n');
fprintf('  Optimal C (BoxConstraint): %.6f\n', mdl_linear.ModelParameters.BoxConstraint);
fprintf('  CV Error: %.4f\n\n', ...
    mdl_linear.HyperparameterOptimizationResults.MinObjective);

fprintf('Nonlinear SVM:\n');
fprintf('  Best Kernel: %s\n', bestKernel);
fprintf('  Optimal C (BoxConstraint): %.6f\n', mdl_nonlinear.ModelParameters.BoxConstraint);
fprintf('  Optimal KernelScale: %.6f\n', mdl_nonlinear.KernelParameters.Scale);
fprintf('  CV Error: %.4f\n\n', bestError);

fprintf('Decision Tree 1 (MaxNumSplits):\n');
fprintf('  Optimal MaxNumSplits: %d\n', mdl_tree_splits.ModelParameters.MaxSplits);
fprintf('  CV Error: %.4f\n\n', ...
    mdl_tree_splits.HyperparameterOptimizationResults.MinObjective);

fprintf('Decision Tree 2 (MinLeafSize):\n');
fprintf('  Optimal MinLeafSize: %d\n', mdl_tree_leaf.ModelParameters.MinLeaf);
fprintf('  CV Error: %.4f\n\n', ...
    mdl_tree_leaf.HyperparameterOptimizationResults.MinObjective);

fprintf('Naive Bayes:\n');
fprintf('  No hyperparameter tuning required.\n\n');

%% SECTION 8: PERFORMANCE METRICS TABLE (Item 2c)
fprintf('--- Item 2c: Model Comparison Summary ---\n\n');

models = {mdl_linear, mdl_nonlinear, mdl_tree_splits, mdl_tree_leaf, mdl_nb};
modelNames = {'Linear SVM', 'Nonlinear SVM', 'DTree (MaxSplits)', 'DTree (MinLeaf)', 'Naive Bayes'};

fprintf('%-25s %10s %10s %10s %10s %10s\n', ...
    'Model', 'Accuracy', 'Precision', 'Recall', 'Specific.', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 75));

cv = cvpartition(Y, 'KFold', 5);

for i = 1:length(models)
    Y_pred = zeros(size(Y));

    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        if isa(models{i}, 'ClassificationSVM')
            temp = fitcsvm(X_scaled(trainIdx,:), Y(trainIdx), ...
                'KernelFunction', models{i}.KernelParameters.Function, ...
                'BoxConstraint', models{i}.ModelParameters.BoxConstraint, ...
                'KernelScale', models{i}.KernelParameters.Scale);
        elseif isa(models{i}, 'ClassificationTree')
            temp = fitctree(X_scaled(trainIdx,:), Y(trainIdx), ...
                'MaxNumSplits', models{i}.ModelParameters.MaxSplits, ...
                'MinLeafSize', models{i}.ModelParameters.MinLeaf);
        elseif isa(models{i}, 'ClassificationNaiveBayes')
            temp = fitcnb(X_scaled(trainIdx,:), Y(trainIdx));
        end

        Y_pred(testIdx) = predict(temp, X_scaled(testIdx,:));
    end

    TP = sum((Y_pred == 1) & (Y == 1));
    TN = sum((Y_pred == 0) & (Y == 0));
    FP = sum((Y_pred == 1) & (Y == 0));
    FN = sum((Y_pred == 0) & (Y == 1));

    accuracy    = (TP + TN) / (TP + TN + FP + FN);
    precision   = TP / (TP + FP);
    recall      = TP / (TP + FN);
    specificity = TN / (TN + FP);
    f1_score    = 2 * (precision * recall) / (precision + recall);

    fprintf('%-25s %10.4f %10.4f %10.4f %10.4f %10.4f\n', ...
        modelNames{i}, accuracy, precision, recall, specificity, f1_score);
end

%% SECTION 9: CONFUSION MATRICES (3x2 GRID)
figure('Name', 'Confusion Matrices', 'Position', [100 100 1200 1200]);

for i = 1:length(models)
    Y_pred = zeros(size(Y));

    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        if isa(models{i}, 'ClassificationSVM')
            temp = fitcsvm(X_scaled(trainIdx,:), Y(trainIdx), ...
                'KernelFunction', models{i}.KernelParameters.Function, ...
                'BoxConstraint', models{i}.ModelParameters.BoxConstraint, ...
                'KernelScale', models{i}.KernelParameters.Scale);
        elseif isa(models{i}, 'ClassificationTree')
            temp = fitctree(X_scaled(trainIdx,:), Y(trainIdx), ...
                'MaxNumSplits', models{i}.ModelParameters.MaxSplits, ...
                'MinLeafSize', models{i}.ModelParameters.MinLeaf);
        elseif isa(models{i}, 'ClassificationNaiveBayes')
            temp = fitcnb(X_scaled(trainIdx,:), Y(trainIdx));
        end

        Y_pred(testIdx) = predict(temp, X_scaled(testIdx,:));
    end

    TP = sum((Y_pred == 1) & (Y == 1));
    TN = sum((Y_pred == 0) & (Y == 0));
    FP = sum((Y_pred == 1) & (Y == 0));
    FN = sum((Y_pred == 0) & (Y == 1));

    subplot(3, 2, i);
    hold on;

    darkBlue = [0.1 0.2 0.5];
    darkRed  = [0.7 0.1 0.1];

    % Row 1: Actual = Diabetes (1)
    fill([0.5 1.5 1.5 0.5], [0.5 0.5 1.5 1.5], darkBlue);  % TP
    fill([1.5 2.5 2.5 1.5], [0.5 0.5 1.5 1.5], darkRed);   % FN
    % Row 2: Actual = No Diabetes (0)
    fill([0.5 1.5 1.5 0.5], [1.5 1.5 2.5 2.5], darkRed);   % FP
    fill([1.5 2.5 2.5 1.5], [1.5 1.5 2.5 2.5], darkBlue);  % TN

    % Cell labels
    text(1, 1, sprintf('True Positive\n(TP)\n%d', TP), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white');
    text(2, 1, sprintf('False Negative\n(FN)\n%d', FN), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white');
    text(1, 2, sprintf('False Positive\n(FP)\n%d', FP), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white');
    text(2, 2, sprintf('True Negative\n(TN)\n%d', TN), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white');

    % X-axis: Predicted
    set(gca, 'XTick', [1 2], ...
        'XTickLabel', {'Diabetes (1)', 'No Diabetes (0)'}, ...
        'FontSize', 9);
    % Y-axis: Actual
    set(gca, 'YTick', [1 2], ...
        'YTickLabel', {'Diabetes (1)', 'No Diabetes (0)'}, ...
        'FontSize', 9);
    set(gca, 'YDir', 'reverse');
    xlim([0.5 2.5]);
    ylim([0.5 2.5]);

    xlabel('Predicted', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Actual', 'FontSize', 11, 'FontWeight', 'bold');

    title(sprintf('%s  (Accuracy: %.2f%%)', modelNames{i}, ...
        (TP+TN)/(TP+TN+FP+FN)*100), 'FontSize', 11);

    hold off;
end

sgtitle('Confusion Matrices - All Models (5-Fold CV)', ...
    'FontSize', 14, 'FontWeight', 'bold');