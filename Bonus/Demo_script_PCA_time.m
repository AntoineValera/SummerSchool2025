function Demo_script_PCA_time(result, behaviour, event_t, smoothing, fadeMemory, fadeAlphaRange)
%DEMO_SCRIPT_PCA_TIME Visualize population activity in PCA space over time.
%
%   Demo_script_PCA_time(result, behaviour, event_t) plots the first three
%   principal components of the neuronal population activity over time and
%   colours the trajectory based on the provided behavioural measure.
%   This is a temporal PCA in which each point of the trajectory corresponds
%   to a time point in neuronal space.
%
%   Inputs
%   ------
%   result     : N x T matrix of neural activity. If empty or omitted the
%                variable is loaded from 'M1_population_data.mat'.
%   behaviour  : 1 x T array describing behaviour at each time point. If
%                empty or omitted it is loaded from the same MAT file.
%   event_t    : Optional indices used when analysing only event times.
%                If omitted or empty and the MAT file contains `event_t`,
%                those indices are used. Otherwise the full time course is
%                displayed.
%   smoothing  : (optional) Gaussian smoothing window in samples. Default 10.
%   fadeMemory : (optional) Number of recent points retained in the dynamic
%                plot. Default 200.
%   fadeAlphaRange : (optional) Two element vector [min max] specifying the
%                transparency range for older to newer points. Default [0.2 1].
%
%   Example
%   -------
%       Demo_script_PCA_time();
%       Demo_script_PCA_time(result, behaviour, [], 5, 150, [0.1 1]);
%
%   This function requires MATLAB's Statistics and Machine Learning
%   Toolbox for the `pca` and `smoothdata` functions.

%   Prior to PCA, neuronal activity is z-scored across time for each neuron
%   so that differences in baseline firing rates or variance do not
%   dominate the principal components. PCA is then performed with each
%   time point treated as an observation in neuronal space, yielding a
%   trajectory through neural state space over time.

    if nargin < 1, result = []; end
    if nargin < 2, behaviour = []; end
    if nargin < 3, event_t = []; end

    if isempty(result) || isempty(behaviour) || isempty(event_t)
        loaded = load('M1_population_data.mat');
        if isfield(loaded, 'mat_data')
            d = loaded.mat_data;
        else
            d = loaded;
        end
        if isempty(result)
            result = d.result;
        end
        if isempty(behaviour)
            behaviour = d.behaviour;
        end
        if isempty(event_t)
            if isfield(d, 'event_t')
                event_t = d.event_t;
            else
                event_t = [];
            end
        end
    end

    if nargin < 4 || isempty(smoothing),   smoothing   = 10;  end
    if nargin < 5 || isempty(fadeMemory),  fadeMemory  = 200; end
    if nargin < 6 || isempty(fadeAlphaRange), fadeAlphaRange = [0.2 1]; end

    % Set analysis options
    normalize_neurons = true; % z-score each neuron across time
    use_peaks         = false;
    speed             = 10; % frame skipping factor

    % remove bad neurons first
    result(all(isnan(result),2), :) = [];

    % Handle NaNs by interpolating over time
    data = fillmissing(result, 'linear', 2, 'EndValues', 'nearest');
    local_beh = behaviour(:)'; % ensure 1 x T row vector
    local_beh = fillmissing(local_beh, 'linear', 'EndValues', 'nearest');

    % Remove time points that still contain NaNs after interpolation
    valid_t = ~any(isnan([data; local_beh]), 1);
    data = data(:, valid_t);
    local_beh = local_beh(valid_t);

    if size(local_beh, 2) ~= size(data, 2)
        error('Behaviour must have the same number of time points as result.');
    end

    % Remove neurons with remaining NaNs
    data = data(~any(isnan(data), 2), :);

    if normalize_neurons
        % Center each neuron's activity across time and scale to unit variance
        % so that all neurons contribute equally to the PCA (fixes issues 1 & 2).
        data = zscore(data, 0, 2);
    end

    % Preprocess calcium and behaviour
    if use_peaks
        speed = 1;
        if ~isempty(event_t)
            data = data(:, event_t);
        end
    end

    mean_calcium = smoothdata(mean(data, 1), 'gaussian', smoothing);

    % Perform PCA
    % Temporal PCA: each time point is an observation, neurons are variables
    [coeff, score] = pca(data'); %#ok<ASGLU>

    score = smoothdata(score, 1, 'gaussian', smoothing);

    % Behaviour array
    sm_behaviour = smoothdata(local_beh, 2, 'gaussian', smoothing*10);
    if use_peaks && ~isempty(event_t)
        sm_behaviour = sm_behaviour(event_t);
    end

    % Prepare for dynamic plotting
    fig = figure;
    set(gcf, 'Color', 'w')
    ax1 = subplot(2,1,1, 'Parent', fig); hold(ax1, 'on');
    view(ax1, 75, 10); % Set azimuth and elevation
    colormap(ax1, 'cool');
    axis equal
    xlabel(ax1, 'PC1'); ylabel(ax1, 'PC2'); zlabel(ax1, 'PC3');
    titleHandle = title(ax1, 'Time: 1');

    ax2 = subplot(2,1,2, 'Parent', fig); hold(ax2, 'on');
    yyaxis(ax2, 'left');
    plot(ax2, mean_calcium, 'Color', [0.6 0.6 0.6]);
    ylabel(ax2, 'Mean calcium');
    yyaxis(ax2, 'right');
    plot(ax2, sm_behaviour, 'b');
    ylabel(ax2, 'Behaviour');
    xlim(ax2, [1 numel(sm_behaviour)]);
    markerLine = xline(ax2, 1, 'k--');
    xlabel(ax2, 'Time');

    fadeAlpha = linspace(fadeAlphaRange(1), fadeAlphaRange(2), fadeMemory);

    % Loop through the time points to update the plot
    for t = 2:speed:size(score, 1)
        cla(ax1);

        if ~use_peaks
            idx = max(1, t-fadeMemory+1):t; % keep only last fadeMemory points
            colours = parula(numel(idx));
            alphas  = fadeAlpha(end-numel(idx)+1:end);
            s = scatter3(ax1, score(idx,1), score(idx,2), score(idx,3), 50, colours, 'filled');
            s.MarkerFaceAlpha = 'flat';
            s.MarkerEdgeAlpha = 'flat';
            s.AlphaData = alphas;
            s.AlphaDataMapping = 'none';
            plot3(ax1, score(t,1), score(t,2), score(t,3), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
        end

        if t > 1
            X = [score(1:t, 1), score(1:t, 1)];
            Y = [score(1:t, 2), score(1:t, 2)];
            Z = [score(1:t, 3), score(1:t, 3)];
            C = repmat(sm_behaviour(1:t)', [1, 2]);
            surface(ax1, X', Y', Z', C', 'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 2);
        end

        markerLine.Value = t;
        titleHandle.String = ['Time: ', num2str(t)];
        drawnow;
    end
end
