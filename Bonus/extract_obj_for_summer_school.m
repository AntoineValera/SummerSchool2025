% load('C:\Users\vanto\Downloads\2019-10-01_exp_1.mat')


% For N neurons and T timepoints : 
% behaviour 	= 1 x T array OR a strcuture of M variables, each one being a 1 x T array
% result 		= N x T matrix
% tax 			= 1 x T array 
% coords 		= N x 3 matrix
% event_t 		= 1 x T2 array. T2 can be used to subsample the signal
use_obj = false

if use_obj
    %% Extract and store extracted variables
    tax 			= obj.t;
    [~, coord_idx] 	= unique(obj.ref.indices.valid_ROIs_list);
    coords 			= obj.ref.simplified_tree{1}.table(coord_idx,3:5);
    event_t 		= vertcat(obj.event.peak_time{:})';
    result 			= obj.rescaled_traces';
    
    %% Extract and store behaviours
    behaviour = struct();
    for el = obj.behaviours.types
        [~, behaviours.(el{1})] = obj.get_activity_bout(el{1});
        behaviour.(el{1}) = behaviours.(el{1}){1};
    end



    save('pop dataset.mat','tax','result','coords','behaviour','event_t','-v7'); % Do not use '-v7.3 if you need to load it with scipy io'
else
    %% Extract and store extracted variables
    tax 			= params.timescale;
    h               = load_header(params.data_folder);
    coords 			= h.original_trees{1}.table(:,3:5);
    event_t 		= [];%vertcat(obj.event.peak_time{:})';
    result 			= squeeze(params.data(:,:,:,:,1,:))';

    %% Extract and store behaviours
    behaviour = struct();
    for el = fieldnames(params.external_var)
        behaviours.(el{1}) = interpolate_to(params.external_var.(el{1}).value, numel(tax));
    end
    save('tree dataset.mat','tax','result','coords','behaviour','event_t','-v7'); % Do not use '-v7.3 if you need to load it with scipy io' 
end

