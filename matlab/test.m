% Copyright 2022 Benjamin Alexander Albert
% All Rights Reserved
% 
% SpartaPlex Academic License
% 
% test.m

function test()

    % read random spheroid center
    fid = fopen("randcenter.txt",'r');
    center = fscanf(fid, "%f");
    fclose(fid);
    
    n = length(center);
    
    % spheroid function handle
    f = @(x) sum((x-center).^2);
    
    fprintf("Optimizing %i-D spheroid...", n);
    start = tic;
    sp = SpartaPlex(n);
    [minvec, minval] = sp.optimize(f);
    fprintf("finished in %0.2f seconds\n", toc(start));

    fprintf("minval = %0.5g\n", minval);

end
