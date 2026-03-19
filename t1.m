Ehr1=readtable("EHRs_1.xlsx");
%% 
%1. Visualizing Missing Data Patterns
Ehr1.VisitDateNum = datenum(Ehr1.VisitDate);%To convert dates to numeric form
Vars1={'ID','VisitDateNum','Age','BMI','Glucose','Cholesterol','BloodPressure','HeartRate','Smoking','Gender_M','Gender_F','Diagnosis_Diabetes','Diagnosis_Hypertension','Diagnosis_HeartDisease'};
numeric=Ehr1{:,Vars1};
missing=isnan(numeric);%Storing the "Nan" values
imagesc(missing);
set(gca, 'XTick', 1:14, 'XTickLabel', Vars1);
xlabel('Variables');
ylabel('Data');
title('Visualization of  NaN Values');
%% 
%2. Quantifying Missing Data
nRows = size(missing, 1);           % Total rows (200)
missingCount = sum(missing, 1);     % Count of NaN per column
missingPercent = (missingCount / nRows) * 100;  

for i = 1:length(Vars1)
     Vars1{i}; missingCount(i); missingPercent(i);
end

% Bar chart of missing percentages
figure;
bar(missingPercent);
set(gca, 'XTick', 1:14, 'XTickLabel', Vars1);
ylabel('Percentage Missing (%)');
xlabel('Variable');
title('Percentage of Missing Values per Variable');

%% 
%3. Imputing Missing Values
% Create subsets
E_original = Ehr1;    % Subset A: Original with missing values (NaN)
E_mean = Ehr1;        % Subset B: Mean imputation
E_median = Ehr1;      % Method 2: Median imputation

fprintf('\n');
fprintf('IMPUTATION: Method 1 (Mean) vs Method 2 (Median)\n');
fprintf('%-15s | %10s | %10s | %8s\n', 'Variable', 'Mean', 'Median', 'N_miss');
fprintf('%s\n', repmat('-', 1, 50));

for i = 1:length(Vars1)
    varName = Vars1{i};
    colData = E_original.(varName);
    nMiss = sum(isnan(colData));
    
    if nMiss > 0
        colMean = mean(colData, 'omitnan');
        colMedian = median(colData, 'omitnan');
        
        % Method 1: Mean imputation
        E_mean.(varName)(isnan(E_mean.(varName))) = colMean;
        
        % Method 2: Median imputation
        E_median.(varName)(isnan(E_median.(varName))) = colMedian;
        
        % PRINT the values - THIS LINE WAS MISSING!
        fprintf('%-15s | %10.2f | %10.2f | %8d\n', varName, colMean, colMedian, nMiss);
    end
end
%% 
% 4. Impact of Missing Data on Analysis
varsToCompare = {'BMI', 'Glucose', 'Cholesterol', 'BloodPressure'};
fprintf('\n');
fprintf('============================================================\n');
fprintf('ITEM 4a: STATISTICAL COMPARISON - Subset A vs Subset B\n');
fprintf('============================================================\n');
fprintf('%-15s | %-28s | %-28s\n', '', 'SUBSET A (Original)', 'SUBSET B (Mean Imputed)');
fprintf('%-15s | %8s %8s %8s | %8s %8s %8s\n', 'Variable', 'Mean', 'Median', 'Std', 'Mean', 'Median', 'Std');
fprintf('%s\n', repmat('-', 1, 75));

% Figure with scatter plots 
figure('Name', 'Impact of Imputation on Distributions', 'Position', [50 100 1400 400]);

for i = 1:length(varsToCompare)
    varName = varsToCompare{i};
    
    % Subset A: Original data (without NaN)
    data_A = E_original.(varName);
    mean_A = mean(data_A, 'omitnan');
    med_A = median(data_A, 'omitnan');
    std_A = std(data_A, 'omitnan');
    n_A = sum(~isnan(data_A));
    
    % Subset B: Mean-imputed data
    data_B = E_mean.(varName);
    mean_B = mean(data_B);
    med_B = median(data_B);
    std_B = std(data_B);
    
    fprintf('%-15s | %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f\n', ...
        varName, mean_A, med_A, std_A, mean_B, med_B, std_B);
    
    % Use scatter plot comparison
    subplot(1, 4, i);
    
    % Plot Original data (without NaN) at x=1
    data_A_valid = data_A(~isnan(data_A));
    scatter(ones(size(data_A_valid)) + 0.1*randn(size(data_A_valid)), data_A_valid, ...
        25, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
    hold on;
    
    % Plot Imputed data at x=2
    scatter(2*ones(size(data_B)) + 0.1*randn(size(data_B)), data_B, ...
        25, 'r', 'filled', 'MarkerFaceAlpha', 0.4);
    
    % Add mean lines
    plot([0.7 1.3], [mean_A mean_A], 'b-', 'LineWidth', 2);
    plot([1.7 2.3], [mean_B mean_B], 'r-', 'LineWidth', 2);
    
    % Add median lines (dashed)
    plot([0.7 1.3], [med_A med_A], 'b--', 'LineWidth', 1.5);
    plot([1.7 2.3], [med_B med_B], 'r--', 'LineWidth', 1.5);
    
    xlim([0.5 2.5]);
    xticks([1 2]);
    xticklabels({'Original (A)', 'Imputed (B)'});
    ylabel(varName, 'FontWeight', 'bold');
    title(sprintf('%s (n=%d→%d)', varName, n_A, length(data_B)), 'FontSize', 10);
   
end
sgtitle('Distribution Comparison: Original vs Imputed (solid=mean, dashed=median)', 'FontSize', 12);


%%
% 5. Detecting & Visualizing Outliers
vars = {'BMI', 'Glucose', 'Cholesterol', 'BloodPressure'};
outlierRowsAll = struct;
maxDeviation = struct;
statsBeforeAfter = struct;

fprintf('\n');
fprintf('============================================================\n');
fprintf('PART 2: OUTLIER DETECTION (using +/- 3 SD method)\n');
fprintf('============================================================\n');

for v = 1:length(vars)
    varName = vars{v};
    x = E_mean.(varName);  % Use imputed data
    
    % Calculate statistics
    mu = mean(x);
    sigma = std(x);
    lowerBound = mu - 3*sigma;
    upperBound = mu + 3*sigma;
    
    % Detect outliers
    outlierIdx = (x > upperBound) | (x < lowerBound);
    outlierRows = find(outlierIdx);
    outlierRowsAll.(varName) = outlierRows;
    
    % Calculate maximum deviation
    if ~isempty(outlierRows)
        deviations = abs(x(outlierIdx) - mu) / sigma;
        maxDeviation.(varName) = max(deviations);
    else
        maxDeviation.(varName) = 0;
    end
    
    % Store before/after statistics
    x_noOut = x(~outlierIdx);
    statsBeforeAfter.(varName).mean_before = mu;
    statsBeforeAfter.(varName).median_before = median(x);
    statsBeforeAfter.(varName).std_before = sigma;
    statsBeforeAfter.(varName).mean_after = mean(x_noOut);
    statsBeforeAfter.(varName).median_after = median(x_noOut);
    statsBeforeAfter.(varName).std_after = std(x_noOut);
    
    % Visualization: Scatter plot with outliers highlighted
    figure('Name', ['Outlier Detection: ' varName]);
    scatter(1:length(x), x, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    if ~isempty(outlierRows)
        scatter(outlierRows, x(outlierIdx), 120, 'r', 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 2);
    end
    
    % Reference lines
    yline(mu, 'g-', 'Mean', 'LineWidth', 2);
    yline(upperBound, 'r--', '+3 SD', 'LineWidth', 1.5);
    yline(lowerBound, 'r--', '-3 SD', 'LineWidth', 1.5);
    
    xlabel('Patient Row Number');
    ylabel(varName);
    title(sprintf('Outlier Detection: %s', varName));
    if ~isempty(outlierRows)
        legend('Normal Values', 'Outliers', 'Location', 'best');
    end
    grid on;
    
    fprintf('%s: Mean=%.2f, SD=%.2f, Bounds=[%.2f, %.2f], Outliers=%d\n', ...
        varName, mu, sigma, lowerBound, upperBound, length(outlierRows));
end


%% 
% 5. Detecting & Visualizing Outliers

% Step 1: Define variables
vars = {'BMI', 'Glucose', 'Cholesterol', 'BloodPressure'};

% Step 2: Initialize structures BEFORE the loop
outlierRowsAll = struct();
maxDeviation = struct();
statsBeforeAfter = struct();

% Step 3: Initialize each field in the structures
for v = 1:length(vars)
    outlierRowsAll.(vars{v}) = [];
    maxDeviation.(vars{v}) = 0;
    statsBeforeAfter.(vars{v}) = struct('mean_before', 0, 'median_before', 0, 'std_before', 0, ...
                                         'mean_after', 0, 'median_after', 0, 'std_after', 0);
end

fprintf('\n');
fprintf('PART 2: OUTLIER DETECTION (using +/- 3 SD method)\n');

% Step 4: Detect outliers for each variable
for v = 1:length(vars)
    varName = vars{v};
    x = E_mean.(varName);  % Use imputed data from Part 1
    
    % Calculate statistics
    mu = mean(x);
    sigma = std(x);
    lowerBound = mu - 3*sigma;
    upperBound = mu + 3*sigma;
    
    % Detect outliers
    outlierIdx = (x > upperBound) | (x < lowerBound);
    outlierRows = find(outlierIdx);
    
    % Store outlier rows
    outlierRowsAll.(varName) = outlierRows;
    
    % Calculate maximum deviation
    if ~isempty(outlierRows)
        deviations = abs(x(outlierIdx) - mu) / sigma;
        maxDeviation.(varName) = max(deviations);
    else
        maxDeviation.(varName) = 0;
    end
    
    % Store before/after statistics
    x_noOut = x(~outlierIdx);
    statsBeforeAfter.(varName).mean_before = mu;
    statsBeforeAfter.(varName).median_before = median(x);
    statsBeforeAfter.(varName).std_before = sigma;
    statsBeforeAfter.(varName).mean_after = mean(x_noOut);
    statsBeforeAfter.(varName).median_after = median(x_noOut);
    statsBeforeAfter.(varName).std_after = std(x_noOut);
    
    % Visualization
    figure('Name', ['Outlier Detection: ' varName]);
    scatter(1:length(x), x, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    if ~isempty(outlierRows)
        scatter(outlierRows, x(outlierIdx), 120, 'r', 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 2);
    end
    yline(mu, 'g-', 'Mean', 'LineWidth', 2);
    yline(upperBound, 'r--', '+3 SD', 'LineWidth', 1.5);
    yline(lowerBound, 'r--', '-3 SD', 'LineWidth', 1.5);
    xlabel('Patient Row Number');
    ylabel(varName);
    title(sprintf('Outlier Detection: %s', varName));
    legend('Normal Values', 'Outliers', 'Location', 'best');
    grid on;
    
    fprintf('%s: Mean=%.2f, SD=%.2f, Outliers=%d\n', varName, mu, sigma, length(outlierRows));
end

%%
% 5a: Report outlier locations
fprintf('\n');
fprintf('ITEM 5a: OUTLIER IDENTIFICATION\n');

for v = 1:length(vars)
    varName = vars{v};
    rows = outlierRowsAll.(varName);
    if ~isempty(rows)
        values = E_mean.(varName)(rows);
        fprintf('%s: %d outlier(s)\n', varName, length(rows));
        fprintf('   Patient rows: %s\n', mat2str(rows'));
        fprintf('   Values: %s\n', mat2str(values', 5));
    else
        fprintf('%s: No outliers detected\n', varName);
    end
end

% Variable with most extreme outliers
devValues = [maxDeviation.BMI, maxDeviation.Glucose, maxDeviation.Cholesterol, maxDeviation.BloodPressure];
[maxDev, maxIdx] = max(devValues);
fprintf('\nVariable with most extreme outlier(s): %s (%.2f SD from mean)\n', ...
    vars{maxIdx}, maxDev);


%% 
% 6. Handle Outliers and Recalculate Statistics
fprintf('\n');
fprintf('============================================================\n');
fprintf('ITEM 6a: STATISTICS BEFORE/AFTER OUTLIER REMOVAL\n');
fprintf('============================================================\n');
fprintf('%-15s | %-24s | %-24s\n', '', 'BEFORE Removal', 'AFTER Removal');
fprintf('%-15s | %7s %7s %7s | %7s %7s %7s\n', 'Variable', 'Mean', 'Median', 'Std', 'Mean', 'Median', 'Std');
fprintf('%s\n', repmat('-', 1, 70));

for v = 1:length(vars)
    varName = vars{v};
    s = statsBeforeAfter.(varName);
    fprintf('%-15s | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f\n', ...
        varName, s.mean_before, s.median_before, s.std_before, ...
        s.mean_after, s.median_after, s.std_after);
end



