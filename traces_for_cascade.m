% Prepare calcium traces for Cascade spike inference.
% Load raw traces from one of the .mat sample. The file should contain a variable named
% 'flattened' with dimensions time x neurons.

load('traces.mat');

% Visualize raw traces
figure; plot(flattened);

% Downsample example data for speed
dF_traces = flattened(1:100:end, :);

% Smooth traces with a Gaussian kernel
dF_traces = smoothdata(dF_traces', 'gaussian', [8 0])';

% Save the processed traces for later use
save('dF_traces.mat', 'dF_traces', '-v7');
