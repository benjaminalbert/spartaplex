% Copyright 2022 Benjamin Alexander Albert
% All Rights Reserved
% 
% SpartaPlex Academic License
% 
% SpartaPlex.m

classdef SpartaPlex < handle
    
    properties
        mesh
        shflstride
        sampstride
    end
    
    methods
        
        % Construct a SpartaPlex optimizer
        % described here: https://doi.org/10.1016/j.advengsoft.2022.103090
        % 
        % Required Args:
        %   n - dimensionality of the search space
        % 
        % Optional Args:
        %   fp32 - (default=false) Boolean indicating mesh precision
        %            true for single-precision (FP32)
        %            false for double-precision (FP64)
        %   gpu  - (default=false) Boolean indicating mesh device
        %            true  to use GPU
        %            false to use CPU
        %            Requires the Parallel Computing Toolbox.
        %            To choose a GPU, call `gpuDevice(device_index)`
        %            before constructing a SpartaPlex optimizer.
        %            Execute `gpuDeviceTable` to view a list of your
        %            accelerators and ensure that the desired device
        %            index indicates an available device.
        %            Note: FP64 computation on most GPUs may be
        %                  significantly slower than FP32.
        function obj = SpartaPlex(n, fp32, gpu)
            if nargin <= 2
                gpu = false;
            end
            if nargin == 1
                fp32 = false;
            end
            obj.mesh = SpartaPlex.makeMesh(n, fp32, gpu);
            obj.shflstride = SpartaPlex.getShflStride(n);
            obj.sampstride = SpartaPlex.getSampStride(n);
        end
        
        % Algorithm 1:
        % SpartaPlex Framework
        % 
        % Required Args:
        %   f - Objective function handle
        %         f must consume an n-by-C matrix and return an array
        %         of 1-by-C evaluations, where n is the dimensionality
        %         provided to the SpartaPlex constructor and
        %         C is no greater than n+1
        % 
        % Optional Args:
        %   maxFE    - (default=n^2) Maximum number of function evaluations
        %   lb       - (default=-ones(n,1)) lower bound vector or scalar
        %                If a vector, it must be of size n-by-1
        %                If a scalar, the bound is applied to all dims
        %   ub       - (default= ones(n,1)) upper bound vector or scalar
        %                If a vector, it must be of size n-by-1
        %                If a scalar, the bound is applied to all dims
        %   printmod - (default=0) Interval at which to print results.
        %                Note: The mesh decay is a function of the number
        %                      of iterations, so printing at an interval of
        %                      1 does not yield a traditional convergence
        %                      curve.
        function [minvec, minval] = optimize(...
            obj, func, maxFE, lb, ub, printmod)
            
            % dimensionality of the mesh and objective function
            n = size(obj.mesh,1);
            
            % make zeroed vector of the same precision
            % and on the same device as the mesh
            minvec = 0*obj.mesh(:,1);
            minval = inf;
            
            if nargin <= 5; printmod =        0; end
            if nargin <= 4; lb       = minvec-1; end
            if nargin <= 3; ub       = minvec+1; end
            if nargin <= 2; maxFE    =      n^2; end
            
            % mesh columns to sample per iteration
            sampcols = 1:obj.sampstride:size(obj.mesh,2);
            crimpedMesh = obj.mesh(:,sampcols);
            
            % number of iterations is the maximum function evaluations
            % divided by the number of vectors evaluated per iteration.
            iterations = floor(maxFE / length(sampcols));
            
            domainmap = @(x) x.*(ub-lb)/2 + (ub+lb)/2;
            for iter=0:iterations-1
                shflsplit = mod(obj.shflstride*iter,n);
                shflsplit = [shflsplit+1:n, 1:shflsplit];
                
                % Section 3.2
                % Algorithm 3 CRIMPS recombinant sampling
                % Cyclic Rotation by Interleaving-Matrix Primality Shuffle
                % and
            	% Section 3.3
            	% Delta(n,I,i) decay function and domain tessellation
                points = min(1, max(-1, minvec + ...
                    (1-2*mod(iter,2)) * (3*n)^(-iter/iterations) * ...
                        crimpedMesh(shflsplit, :)));
                [iterMinVal, iterMinIdx] = min(func(domainmap(points)));
                if (iterMinVal < minval)
                    minvec = points(:,iterMinIdx);
                    minval = iterMinVal;
                end
                if (printmod && ~mod(iter+1, printmod))
                    fprintf("iter %i/%i minval %0.5g\n", ...
                        iter+1, iterations, minval);
                end
            end
            minvec = domainmap(minvec);
        end
    end
    
    methods(Static)
        
        % Section 3.1 Algorithm 2:
        % M mesh generation
        % See the SpartaPlex constructor documentation.
        function M = makeMesh(n, fp32, gpu)
            if fp32
                fptype = "single";
            else
                fptype = "double";
            end
            M = [[1 -ones(1, n, fptype)/n] ; zeros(n-1, n+1, fptype)];
            for x=2:n
                M(x,x) = sqrt(1 - dot(M(:,x), M(:,x)));
                M(x, x+1:n+1) = -(1/n + dot(M(:,x), M(:,x+1))) / M(x,x);
            end
            if gpu
                M = gpuArray(M);
            end
        end
        
        % Calculate CRIMPS prime, used as the shflstride.
        % CRIMPS prime is the smallest prime in the range [floor(n/2)+1,n]
        function shflstride = getShflStride(n)
            p = primes(n);
            shflstride = min(p(p > (n/2)));
        end

        % Calculate the mesh sampling stride.
        % sample stride is the first odd value >= log2(n)
        % or 1 if n < 10
        function sampstride = getSampStride(n)
            if n < 10
                sampstride = 1;
            else
                sampstride = 2*floor(log2(n)/2)+1;
            end
        end
        
        % unit test function to check that:
        % 1. the mesh columns are unit vectors
        % 2. every dihedral angle is equal to arccos(-1/n)
        % 
        % if ret is true, then the mesh
        % columns come from a unit regular simplex
        % otherwise, the mesh is invalid.
        % 
        % Required Args:
        %   M - a mesh to evaluate for validity
        % 
        % Optional Args:
        %   tol - (default=1e-5) tolerance for rounding errors
        function ret = validateMesh(M, tol)
            if nargin==1
                tol = 1e-5;
            end
            ret = true;
            if any(abs(vecnorm(M) - 1) > tol)
                ret = false;
            else
                for c=1:size(M,2)-1
                    dihedralAngles = acos(sum(M(:,c).*M(:,c+1:size(M,2))));
                    if any(abs(dihedralAngles - acos(-1/size(M,1))) > tol)
                        ret = false;
                        break
                    end
                end
            end
        end
    end
end
