clear;
clc;
rng(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BME 516/616 - ASSIGNMENT #4
%% PART 1: DIMENSIONALITY REDUCTION & CLUSTERING
%% Dataset: Preprocessed_Gene_Dataset.mat (599 obs, 7 features)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ========================================================================
%% SECTION 1: PRINCIPAL COMPONENT ANALYSIS (PCA)
%% Items 1a, 1b, 1c
%% ========================================================================

% ----- Load Dataset ------------------------------------------------------
geneData = load('Preprocessed_Gene_Dataset.mat');
X = geneData.Preprocessed_Gene_Dataset;

% ----- Run PCA (default settings) ----------------------------------------
[coeff, score, latent] = pca(X);

fprintf('=== PCA RESULTS ===\n');
fprintf('Total Principal Components Generated: %d\n', length(latent));

% ----- Variance Calculations ---------------------------------------------
explainedVar       = latent / sum(latent);
cumulativeVariance = cumsum(explainedVar);

fprintf('\n=== VARIANCE EXPLAINED PER PC ===\n');
for i = 1:length(latent)
    fprintf('PC%d: %.2f%% | Cumulative: %.2f%%\n', ...
        i, explainedVar(i)*100, cumulativeVariance(i)*100);
end

% ----- Figure 1: Scree Plot + Cumulative Variance (Items 1a & 1b) --------
figure('Name', 'Figure 1: PCA Variance Analysis', ...
       'Position', [100 100 1200 500]);

subplot(1, 2, 1);
bar(1:length(latent), explainedVar * 100, 'FaceColor', [0.2 0.5 0.8]);
title('Scree Plot', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Principal Component', 'FontSize', 11);
ylabel('Variance Explained (%)', 'FontSize', 11);
grid on;

subplot(1, 2, 2);
plot(1:length(latent), cumulativeVariance * 100, '-o', ...
    'LineWidth', 2, 'MarkerFaceColor', 'b', 'Color', [0.2 0.5 0.8]);
yline(95, '--r', '95% Threshold', 'LabelHorizontalAlignment', 'left');
title('Cumulative Variance Explained', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Number of Principal Components', 'FontSize', 11);
ylabel('Cumulative Variance (%)', 'FontSize', 11);
ylim([0 105]);
grid on;

sgtitle('Figure 1: PCA Variance Analysis (Items 1a & 1b)', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% SECTION 2: K-MEANS CLUSTERING & CLUSTER EVALUATION
%% Items 2a, 2b, 2c, 2d, 2e
%% ========================================================================

% ----- Extract First 3 PCs (defined HERE before everything that needs it)
score3D = score(:, 1:3); % <-- MOVED UP: must be defined before for loop AND evalclusters

% ----- Figure 2: 3D Scatter - No Clustering Assumed (Item 1c) ------------
figure('Name', 'Figure 2: 3D Scatter No Clustering', ...
       'Position', [100 100 700 600]);

scatter3(score3D(:,1), score3D(:,2), score3D(:,3), 15, 'b', 'filled');
title({'3D Scatter: First 3 Principal Components', ...
       '(Rotate to visually inspect for clusters)'}, ...
       'FontSize', 12, 'FontWeight', 'bold');
xlabel('PC 1', 'FontSize', 11);
ylabel('PC 2', 'FontSize', 11);
zlabel('PC 3', 'FontSize', 11);
grid on;
view(45, 30);

% ----- K-Means For Loop: K = 1 to 10 (Items 2b & 2c) --------------------
maxK = 10;
silhouetteScores = zeros(1, maxK);

figure('Name', 'Figure 3: K-Means Clustering K=1 to 10', ...
       'Position', [0 0 1800 800]);

for k = 1:maxK

    % Perform K-Means (Replicates=5 for stability)
    idx = kmeans(score3D, k, 'Replicates', 5);

    % Silhouette score undefined for k=1
    if k > 1
        s = silhouette(score3D, idx);
        silhouetteScores(k) = mean(s);
    else
        silhouetteScores(k) = NaN;
    end

    % 2x5 grid subplot per K value
    subplot(2, 5, k);
    scatter3(score3D(:,1), score3D(:,2), score3D(:,3), 8, idx, 'filled');
    title(['K = ' num2str(k) '  |  Sil: ' num2str(round(silhouetteScores(k),3))], ...
          'FontSize', 9, 'FontWeight', 'bold');
    xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
    view(45, 30);
    grid on;

end

sgtitle('Figure 3: K-Means Clustering Comparison K = 1 to 10 (Items 2b & 2c)', ...
        'FontSize', 14, 'FontWeight', 'bold');

% ----- Figure 4: Silhouette & Calinski-Harabasz Plots (Items 2b & 2d) ----
evalResults = evalclusters(score3D, 'kmeans', 'CalinskiHarabasz', ...
                           'KList', 1:maxK);

figure('Name', 'Figure 4: Cluster Evaluation Metrics', ...
       'Position', [100 100 1200 500]);

subplot(1, 2, 1);
plot(2:maxK, silhouetteScores(2:end), '-ro', ...
    'MarkerFaceColor', 'r', 'LineWidth', 2);
title('Silhouette Score vs. Number of Clusters', ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('Number of Clusters (k)', 'FontSize', 11);
ylabel('Average Silhouette Score', 'FontSize', 11);
grid on;

subplot(1, 2, 2);
plot(evalResults.InspectedK, evalResults.CriterionValues, '-bo', ...
    'MarkerFaceColor', 'b', 'LineWidth', 2);
title('Calinski-Harabasz Criterion vs. Number of Clusters', ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('Number of Clusters (k)', 'FontSize', 11);
ylabel('Calinski-Harabasz Index', 'FontSize', 11);
grid on;

sgtitle('Figure 4: Cluster Evaluation Metrics (Items 2b & 2d)', ...
        'FontSize', 14, 'FontWeight', 'bold');

% ----- Print Optimal K Results -------------------------------------------
[~, bestK_sil] = max(silhouetteScores(2:end));
bestK_sil = bestK_sil + 1;

fprintf('\n=== OPTIMAL CLUSTER RESULTS ===\n');
fprintf('Optimal k (Silhouette Score):  k = %d (Score = %.4f)\n', ...
        bestK_sil, silhouetteScores(bestK_sil));
fprintf('Optimal k (Calinski-Harabasz): k = %d\n', evalResults.OptimalK);