function my_pool=Create_Pool(n_workers)

poolObj = gcp('nocreate');   % Get current pool
if ~isempty(poolObj)
    delete(poolObj);
    fprintf('Existing parallel pool closed.\n');
else
    fprintf('No active parallel pool.\n');
end
if n_workers == 1
    my_pool = parpool("threads",1);  % extremely light‑weight
else
    % Retry logic for starting a new parpool
    maxRetries = 10;
    for attempt = 1:maxRetries
        try
            delete(gcp('nocreate'));    % Close any existing pool
            ps = parallel.Settings;
            ps.Pool.AutoCreate  = false;  % Disable auto-creation on parfor
            ps.Pool.IdleTimeout = Inf;    % Disable idle timeout

            fprintf('Attempt %d/%d: Starting pool...\n', attempt, maxRetries);
            my_pool=parpool(n_workers);
            fprintf('Parallel pool started successfully.\n');
            break;
        catch ME
            warning('Attempt %d failed: %s', attempt, ME.message);
            if attempt == maxRetries
                rethrow(ME);
            else
                pause(2);  % Wait before retry
            end
        end
    end
end
end