clear; clc; rng(1);

%% =========================================================
%% 1. DATA LOADING & CLEANING
%% =========================================================
filename = 'AD Data Set #2.xlsx';
opts = detectImportOptions(filename);
data = readtable(filename, opts);
labelColumn = 'any_impaired_dx';

isNumeric = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
excludeCols = {'cnsa_id', 'adrc_id', 'date_of_blood_collection', 'studies_enrolled'};
isNotID = ~ismember(data.Properties.VariableNames, excludeCols) & ...
          ~strcmp(data.Properties.VariableNames, labelColumn);

X_raw  = data{:, isNumeric & isNotID};
Y      = categorical(data.(labelColumn));
biomarkerNames = data.Properties.VariableNames(isNumeric & isNotID);

X_cleaned = fillmissing(X_raw, 'knn');
X_scaled  = normalize(X_cleaned);

%% =========================================================
%% 2. FEATURE SELECTION  –  ReliefF (top 15)
%% =========================================================
[rankedIdx, rel_weights] = relieff(X_scaled, Y, 10);
X_selected    = X_scaled(:, rankedIdx(1:15));
selectedNames = biomarkerNames(rankedIdx(1:15));
fprintf('Using top 15 ReliefF features — no PCA\n');

%% =========================================================
%% 3. HELPER FUNCTIONS
%% =========================================================
function [X_out, Y_out] = applySmote(X_in, Y_in, k)
    idxNo = find(Y_in == 'No');
    X_no  = X_in(idxNo, :);
    nGen  = sum(Y_in == 'Yes') - sum(Y_in == 'No');
    if nGen <= 0, X_out = X_in; Y_out = Y_in; return; end
    X_syn = zeros(nGen, size(X_in, 2));
    for i = 1:nGen
        ri   = randi(size(X_no, 1));
        root = X_no(ri, :);
        d    = sum((X_no - root).^2, 2);
        [~, s] = sort(d);
        nb   = X_no(s(randi([2, k+1])), :);
        X_syn(i,:) = root + rand * (nb - root);
    end
    X_out = [X_in;  X_syn];
    Y_out = [Y_in;  repmat(categorical({'No'}), nGen, 1)];
end

function [acc, prec, rec, f1, spec, npv, mcc] = allMetrics(actual, pred, posClass)
    TP = sum(actual == posClass & pred == posClass);
    TN = sum(actual ~= posClass & pred ~= posClass);
    FP = sum(actual ~= posClass & pred == posClass);
    FN = sum(actual == posClass & pred ~= posClass);
    acc  = (TP + TN) / (TP + TN + FP + FN + eps) * 100;
    prec = TP / (TP + FP + eps) * 100;
    rec  = TP / (TP + FN + eps) * 100;
    f1   = 2*TP / (2*TP + FP + FN + eps) * 100;
    spec = TN / (TN + FP + eps) * 100;
    npv  = TN / (TN + FN + eps) * 100;
    mcc  = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + eps);
end

%% =========================================================
%% 4. CV LOOP  –  SMOTE inside each fold 
%% =========================================================
K  = 5;
cv = cvpartition(Y, 'KFold', K, 'Stratify', true);

predsRF   = categorical(repmat({''}, size(Y)));
predsSVM  = categorical(repmat({''}, size(Y)));
predsNB   = categorical(repmat({''}, size(Y)));
scoresRF  = zeros(size(Y));
scoresSVM = zeros(size(Y));
scoresNB  = zeros(size(Y));

for fold = 1:K
    trIdx = training(cv, fold);
    teIdx = test(cv,     fold);

    X_tr = X_selected(trIdx, :);  Y_tr = Y(trIdx);
    X_te = X_selected(teIdx, :);

    % SMOTE on training fold only
    [X_tr_aug, Y_tr_aug] = applySmote(X_tr, Y_tr, 5);

    % ── Random Forest ──────────────────────────────────
    mdlRF = fitcensemble(X_tr_aug, Y_tr_aug, ...
        'Method', 'Bag', 'NumLearningCycles', 150);
    [predsRF(teIdx), postRF] = predict(mdlRF, X_te);
    yesColRF = find(mdlRF.ClassNames == categorical({'Yes'}));
    scoresRF(teIdx) = postRF(:, yesColRF);

    % ── RBF-SVM (Platt scaling) ─────────────────────────
    mdlSVM = fitcsvm(X_tr_aug, Y_tr_aug, ...
        'KernelFunction', 'rbf', 'Standardize', true, ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', ...
         struct('ShowPlots', false, 'Verbose', 0, ...
                'MaxObjectiveEvaluations', 20));
    mdlSVM = fitPosterior(mdlSVM);
    [predsSVM(teIdx), postSVM] = predict(mdlSVM, X_te);
    yesColSVM = find(mdlSVM.ClassNames == categorical({'Yes'}));
    scoresSVM(teIdx) = postSVM(:, yesColSVM);

    % ── Naïve Bayes ─────────────────────────────────────
    mdlNB = fitcnb(X_tr_aug, Y_tr_aug, 'DistributionNames', 'kernel');
    [predsNB(teIdx), postNB] = predict(mdlNB, X_te);
    yesColNB = find(mdlNB.ClassNames == categorical({'Yes'}));
    scoresNB(teIdx) = postNB(:, yesColNB);
end

%% =========================================================
%% 5. COMPUTE ALL METRICS
%% =========================================================
posClass = categorical({'Yes'});
[accRF,  precRF,  recRF,  f1RF,  specRF,  npvRF,  mccRF ] = allMetrics(Y, predsRF,  posClass);
[accSVM, precSVM, recSVM, f1SVM, specSVM, npvSVM, mccSVM] = allMetrics(Y, predsSVM, posClass);
[accNB,  precNB,  recNB,  f1NB,  specNB,  npvNB,  mccNB ] = allMetrics(Y, predsNB,  posClass);

[fprRF,  tprRF,  ~, aucRF ] = perfcurve(Y, scoresRF,  'Yes');
[fprSVM, tprSVM, ~, aucSVM] = perfcurve(Y, scoresSVM, 'Yes');
[fprNB,  tprNB,  ~, aucNB ] = perfcurve(Y, scoresNB,  'Yes');

%% =========================================================
%% 6. DASHBOARD
%% =========================================================
C.bg      = [0.06 0.08 0.12];
C.panel   = [0.10 0.13 0.19];
C.accent1 = [0.25 0.72 0.85];
C.accent2 = [0.98 0.60 0.18];
C.accent3 = [0.35 0.88 0.55];
C.danger  = [0.95 0.32 0.32];
C.textH   = [0.95 0.97 1.00];
C.textS   = [0.65 0.72 0.82];

fig = figure('Name','SaMD – Early ADRD Cognitive Screening', ...
    'Color',C.bg,'Position',[30 20 1600 1000], ...
    'ToolBar','none','MenuBar','none');

% ── Header ─────────────────────────────────────────────────
annotation('textbox',[0 0.955 1 0.045], ...
    'String','SaMD  ■  Early Cognitive Change Detection in ADRD', ...
    'FontName','Consolas','FontSize',16,'FontWeight','bold', ...
    'Color',C.accent1,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor',C.bg);
annotation('textbox',[0 0.928 1 0.028], ...
    'String',['Blood Biomarker Panel  ·  ReliefF(15)  ·  ' ...
              'SMOTE in-fold  ·  5-Fold Stratified CV  ·  Platt ROC/AUC'], ...
    'FontName','Consolas','FontSize',8.5,'Color',C.textS,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor',C.bg);

% ── Panel A: Biomarker Importance ──────────────────────────
ax1 = subplot('Position',[0.04 0.58 0.18 0.32]);
topW = rel_weights(rankedIdx(1:10));
cmap = [linspace(C.accent1(1),C.accent3(1),10)', ...
        linspace(C.accent1(2),C.accent3(2),10)', ...
        linspace(C.accent1(3),C.accent3(3),10)'];
barh(1:10, topW(end:-1:1), 0.65, 'FaceColor','flat','CData',cmap);
set(ax1,'Color',C.panel,'XColor',C.textS,'YColor',C.textS, ...
    'FontName','Consolas','FontSize',8, ...
    'YTick',1:10,'YTickLabel',flip(selectedNames(1:10)), ...
    'GridColor',[1 1 1 0.08],'XGrid','on');
title(ax1,'A.  Biomarker Importance (ReliefF)', ...
    'Color',C.textH,'FontName','Consolas','FontSize',10);
xlabel(ax1,'ReliefF Weight','Color',C.textS,'FontName','Consolas');

% ── Panel B: SMOTE Class Balancing ─────────────────────────
ax2 = subplot('Position',[0.25 0.58 0.16 0.32]);
groups   = categorical({'Impaired','Healthy'},{'Impaired','Healthy'});
origCnts = [sum(Y=='Yes'), sum(Y=='No')];
augCnts  = [sum(Y=='Yes'), sum(Y=='No')+(sum(Y=='Yes')-sum(Y=='No'))];
hold(ax2,'on');
bar(ax2,groups(1),augCnts(1),  0.6,'FaceColor',C.accent2,'FaceAlpha',0.35,'EdgeColor','none');
bar(ax2,groups(2),augCnts(2),  0.6,'FaceColor',C.accent1,'FaceAlpha',0.35,'EdgeColor','none');
bar(ax2,groups(1),origCnts(1), 0.4,'FaceColor',C.accent2,'EdgeColor','none');
bar(ax2,groups(2),origCnts(2), 0.4,'FaceColor',C.accent1,'EdgeColor','none');
set(ax2,'Color',C.panel,'XColor',C.textS,'YColor',C.textS, ...
    'FontName','Consolas','FontSize',8, ...
    'GridColor',[1 1 1 0.08],'YGrid','on');
title(ax2,'B.  SMOTE Class Balancing', ...
    'Color',C.textH,'FontName','Consolas','FontSize',10);
ylabel(ax2,'N Patients','Color',C.textS,'FontName','Consolas');
legend(ax2,{'Post-SMOTE','Pre-SMOTE'}, ...
    'TextColor',C.textS,'Color',C.panel,'EdgeColor','none', ...
    'FontName','Consolas','FontSize',7);
text(ax2,1,origCnts(1)+2,num2str(origCnts(1)), ...
    'HorizontalAlignment','center','Color',C.textH,'FontName','Consolas','FontSize',9);
text(ax2,2,origCnts(2)+2,num2str(origCnts(2)), ...
    'HorizontalAlignment','center','Color',C.textH,'FontName','Consolas','FontSize',9);

% ── Panel C: Core 5-Metric Grouped Bar ─────────────────────
ax3 = subplot('Position',[0.44 0.58 0.27 0.32]);
metricLabels = {'Accuracy','Precision','Recall','F1-Score','Specificity'};
vRF  = [accRF,  precRF,  recRF,  f1RF,  specRF ];
vSVM = [accSVM, precSVM, recSVM, f1SVM, specSVM];
vNB  = [accNB,  precNB,  recNB,  f1NB,  specNB ];
bw = 0.25;
bar(ax3,(1:5)-bw, vRF,  bw,'FaceColor',C.accent3,'EdgeColor','none');
hold(ax3,'on');
bar(ax3, 1:5,     vSVM, bw,'FaceColor',C.accent1,'EdgeColor','none');
bar(ax3,(1:5)+bw, vNB,  bw,'FaceColor',C.accent2,'EdgeColor','none');
set(ax3,'Color',C.panel,'XColor',C.textS,'YColor',C.textS, ...
    'XTick',1:5,'XTickLabel',metricLabels,'FontName','Consolas','FontSize',8, ...
    'GridColor',[1 1 1 0.08],'YGrid','on','YLim',[0 110]);
title(ax3,'C.  Accuracy · Precision · Recall · F1 · Specificity', ...
    'Color',C.textH,'FontName','Consolas','FontSize',10);
ylabel(ax3,'Score (%)','Color',C.textS,'FontName','Consolas');
legend(ax3,{'RF','SVM','NB'}, ...
    'TextColor',C.textS,'Color',C.panel,'EdgeColor','none', ...
    'FontName','Consolas','FontSize',8,'Location','southeast');

% ── Panel D: ROC Curves ────────────────────────────────────
ax4 = subplot('Position',[0.74 0.58 0.24 0.32]);
hold(ax4,'on');
plot(ax4,[0 1],[0 1],'--','Color',[1 1 1 0.2],'LineWidth',1);
fill(ax4,[fprRF;  1;0],[tprRF;  0;0],C.accent3,'FaceAlpha',0.08,'EdgeColor','none');
fill(ax4,[fprSVM; 1;0],[tprSVM; 0;0],C.accent1,'FaceAlpha',0.08,'EdgeColor','none');
fill(ax4,[fprNB;  1;0],[tprNB;  0;0],C.accent2,'FaceAlpha',0.08,'EdgeColor','none');
plot(ax4,fprRF,  tprRF,  '-', 'Color',C.accent3,'LineWidth',2.5);
plot(ax4,fprSVM, tprSVM, '-', 'Color',C.accent1,'LineWidth',2.5);
plot(ax4,fprNB,  tprNB,  '--','Color',C.accent2,'LineWidth',1.8);
[~,iRF]  = min(sqrt(fprRF.^2  + (1-tprRF).^2));
[~,iSVM] = min(sqrt(fprSVM.^2 + (1-tprSVM).^2));
[~,iNB]  = min(sqrt(fprNB.^2  + (1-tprNB).^2));
plot(ax4,fprRF(iRF),   tprRF(iRF),   'o','Color',C.accent3,'MarkerSize',7,'MarkerFaceColor',C.accent3);
plot(ax4,fprSVM(iSVM), tprSVM(iSVM), 'o','Color',C.accent1,'MarkerSize',7,'MarkerFaceColor',C.accent1);
plot(ax4,fprNB(iNB),   tprNB(iNB),   'o','Color',C.accent2,'MarkerSize',7,'MarkerFaceColor',C.accent2);
text(ax4,0.52,0.22,sprintf('AUC = %.3f',aucRF), 'Color',C.accent3,'FontName','Consolas','FontSize',9,'FontWeight','bold');
text(ax4,0.52,0.14,sprintf('AUC = %.3f',aucSVM),'Color',C.accent1,'FontName','Consolas','FontSize',9,'FontWeight','bold');
text(ax4,0.52,0.06,sprintf('AUC = %.3f',aucNB), 'Color',C.accent2,'FontName','Consolas','FontSize',9,'FontWeight','bold');
set(ax4,'Color',C.panel,'XColor',C.textS,'YColor',C.textS, ...
    'FontName','Consolas','FontSize',9, ...
    'GridColor',[1 1 1 0.08],'XGrid','on','YGrid','on','XLim',[0 1],'YLim',[0 1]);
title(ax4,'D.  ROC-AUC  (Platt-Calibrated)', ...
    'Color',C.textH,'FontName','Consolas','FontSize',10);
xlabel(ax4,'False Positive Rate','Color',C.textS,'FontName','Consolas');
ylabel(ax4,'True Positive Rate','Color',C.textS,'FontName','Consolas');
legend(ax4,{'Chance','','','','RF','SVM','NB'}, ...
    'TextColor',C.textS,'Color',C.panel,'EdgeColor','none', ...
    'FontName','Consolas','FontSize',8,'Location','southeast');

% ── Panels E–G: Confusion Matrices ─────────────────────────
models     = {predsRF,            predsSVM,      predsNB};
mTitles    = {'E.  Random Forest','F.  RBF-SVM','G.  Naïve Bayes'};
accs       = {accRF,  accSVM,  accNB};
aucs       = {aucRF,  aucSVM,  aucNB};
xPos       = [0.04  0.36  0.68];
classLabels = {'Yes','No'};

for m = 1:3
    ax = axes(fig,'Position',[xPos(m) 0.05 0.28 0.44]);
    actual = models{m};
    CM = zeros(2,2);
    for r = 1:2
        for c = 1:2
            CM(r,c) = sum(actual == categorical({classLabels{c}}) & ...
                          Y      == categorical({classLabels{r}}));
        end
    end
    hold(ax,'on');
    for r = 1:2
        for c = 1:2
            faceCol = C.danger;
            if r == c, faceCol = C.accent3; end
            fill(ax,[c-1 c c c-1],[r-1 r-1 r r],faceCol, ...
                'FaceAlpha',0.45,'EdgeColor',C.bg,'LineWidth',2);
            text(ax,c-0.5,r-0.5,num2str(CM(r,c)), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontName','Consolas','FontSize',24,'FontWeight','bold','Color',C.textH);
        end
    end
    set(ax,'Color',C.panel,'XColor',C.textH,'YColor',C.textH, ...
        'FontName','Consolas','FontSize',10, ...
        'XTick',[0.5 1.5],'XTickLabel',classLabels, ...
        'YTick',[0.5 1.5],'YTickLabel',classLabels, ...
        'XLim',[0 2],'YLim',[0 2],'XAxisLocation','bottom','TickLength',[0 0]);
    ax.YDir = 'reverse';
    xlabel(ax,'Predicted','Color',C.textS,'FontName','Consolas','FontSize',10);
    ylabel(ax,'True',     'Color',C.textS,'FontName','Consolas','FontSize',10);
    title(ax,sprintf('%s  |  Acc %.1f%%  ·  AUC %.3f', ...
        mTitles{m},accs{m},aucs{m}), ...
        'Color',C.textH,'FontName','Consolas','FontSize',10);
    text(ax,0.5,0.15,'TP','Color',C.accent3,'FontName','Consolas','FontSize',8,'HorizontalAlignment','center');
    text(ax,1.5,1.15,'TN','Color',C.accent3,'FontName','Consolas','FontSize',8,'HorizontalAlignment','center');
    text(ax,1.5,0.15,'FN','Color',C.danger, 'FontName','Consolas','FontSize',8,'HorizontalAlignment','center');
    text(ax,0.5,1.15,'FP','Color',C.danger, 'FontName','Consolas','FontSize',8,'HorizontalAlignment','center');
end

%% =========================================================
%% 7. CONSOLE REPORT
%% =========================================================
fprintf('\n╔══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║           SaMD PERFORMANCE REPORT  –  ADRD Screening                ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Pipeline: ReliefF(15) → SMOTE-in-fold → 5-Fold Stratified CV       ║\n');
fprintf('║  Top-3 Predictors: %-50s║\n', ...
    sprintf('%s, %s, %s',selectedNames{1},selectedNames{2},selectedNames{3}));
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ %-14s %6s %6s %6s %6s %6s %6s %6s %6s ║\n', ...
    'Model','Acc','Prec','Recall','F1','Spec','NPV','MCC','AUC');
fprintf('╠══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ %-14s %5.1f%% %5.1f%% %5.1f%%  %5.1f%% %5.1f%% %5.1f%% %5.3f %5.3f ║\n', ...
    'Random Forest', accRF,  precRF,  recRF,  f1RF,  specRF,  npvRF,  mccRF,  aucRF);
fprintf('║ %-14s %5.1f%% %5.1f%% %5.1f%%  %5.1f%% %5.1f%% %5.1f%% %5.3f %5.3f ║\n', ...
    'RBF-SVM',       accSVM, precSVM, recSVM, f1SVM, specSVM, npvSVM, mccSVM, aucSVM);
fprintf('║ %-14s %5.1f%% %5.1f%% %5.1f%%  %5.1f%% %5.1f%% %5.1f%% %5.3f %5.3f ║\n', ...
    'Naive Bayes',   accNB,  precNB,  recNB,  f1NB,  specNB,  npvNB,  mccNB,  aucNB);
fprintf('╚══════════════════════════════════════════════════════════════════════╝\n');
fprintf('NOTE: SMOTE fitted inside CV folds — no data leakage.\n\n');