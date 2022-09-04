# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert
All Rights Reserved

SpartaPlex Academic License

spartaplex.py
------------------------------------------------------------------------
"""

import torch


class SpartaPlex:

    def __init__(
            self,
            n,
            fp32=False,
            dev="cpu"):
        """
        Construct a SpartaPlex optimizer
        described here: https://doi.org/10.1016/j.advengsoft.2022.103090
        
        Required Args
        -------------

        n : int
            dimensionality of the search space
        
        Optional Args
        -------------

        fp32 : bool (default=false)
            Boolean indicating mesh precision
            True for single-precision (FP32)
            False for double-precision (FP64)

        dev : str,int (default="cpu")
            PyTorch device for all computation
            "cpu" to indicate CPU
            specify the device index to use a GPU
        """
        self.mesh = SpartaPlex.makeMesh(n, fp32, dev)
        self.shflstride = SpartaPlex.getShflStride(n)
        self.sampstride = SpartaPlex.getSampStride(n)

    def optimize(
            self,
            objfunc,
            maxFE=None,
            lb=-1,
            ub=1,
            printmod=0):
        """
        Algorithm 1:
        SpartaPlex Framework
        
        Required Args
        -------------

        f : Callable
            Objective function handle
            f must consume an n-by-C matrix and return an array
            of 1-by-C evaluations, where n is the dimensionality
            provided to the SpartaPlex constructor and
            C is no greater than n+1
        
        Optional Args
        -------------

        maxFE : int (default=n^2)
            Maximum number of function evaluations

        lb : float,tensor (default=-1)
            lower bound vector or scalar
            If a scalar, the bound is applied to all dims
            If a vector, it must be of size n-by-1

        ub : float,tensor (default=1)
            upper bound vector or scalar
            If a scalar, the bound is applied to all dims
            If a vector, it must be of size n-by-1

        printmod : int (default=0)
            Interval at which to print results.
            Note: The mesh decay is a function of the number
            of iterations, so printing at an interval of
            1 does not yield a traditional convergence curve.

        Returns
        -------
        Tuple with elements: (minvec, minval)
            where minvec is the Tensor that minimizes the objective function
            and minval is the evaluation of the objective function at minvec
        """

        # dimensionality of the mesh and objective function
        n = self.mesh.shape[0]

        if maxFE is None:
            maxFE = n**2

        # make zeroed vector of the same precision
        # and on the same device as the mesh
        minvec = 0*self.mesh[:,0].unsqueeze(1)
        minval = torch.inf

        # mesh columns to sample per iteration
        sampcols = torch.tensor(
            data=[x for x in range(0, n+1, self.sampstride)],
            device=self.mesh.device)
        crimpedMesh = torch.index_select(self.mesh, 1, sampcols)

        # number of iterations is the maximum function evaluations
        # divided by the number of vectors evaluated per iteration.
        iterations = (maxFE // len(sampcols))

        def _domainMap(x):
            return x * (ub-lb)/2 + (ub+lb)/2

        for i in range(iterations):
            shflsplit = (self.shflstride*i) % n
            shflsplit = torch.tensor(
                data=[(x + shflsplit) % n for x in range(n)],
                device=self.mesh.device)
            # Section 3.2 Algorithm 3 CRIMPS recombinant sampling
            # Cyclic Rotation by Interleaving-Matrix Primality Shuffle
            points = torch.clip(
            	# Section 3.3
            	# Delta(n,I,i) decay function and domain tessellation
                input=minvec + (1-2*(i%2)) * ((3*n)**(-i/iterations)) *
                    torch.index_select(crimpedMesh, 0, shflsplit),
                min=-1,
                max=1)
            (iterMinVal, iterMinIdx) = torch.min(
                input=objfunc(_domainMap(points)),
                dim=0)
            if iterMinVal < minval:
                minval = iterMinVal
                minvec = points[:, iterMinIdx].unsqueeze(1)
            if printmod and not ((i+1) % printmod):
                print("iter {}/{} minval {:0.5g}".format(
                    i+1, iterations, minval))

        return _domainMap(minvec), minval

    @staticmethod
    def makeMesh(n, fp32, dev):
        """
        Section 3.1 Algorithm 2:
        M mesh generation

        See the SpartaPlex constructor documentation
        for information about the arguments.

        Returns
        -------
        Tensor of shape [n,n+1] representing a unit n-simplex.
        """
        M = torch.zeros(
            size=(n,n+1),
            dtype=(torch.float32 if fp32 else torch.float64),
            device=dev)
        M[0,0  ] =  1
        M[0,1:n] = -1/n
        for r in range(1,n):
            M[r,r    ] = torch.sqrt(1-torch.sum(torch.square(M[:,r])))
            M[r,r+1:n] = -(1/n + torch.dot(M[:,r], M[:,r+1])) / M[r,r]
        M[:,n] = M[:,n-1]
        M[n-1,n] = -M[n-1,n-1]
        return M

    @staticmethod
    def getShflStride(n):
        """
        Calculate CRIMPS prime, used as the shflstride.
        CRIMPS prime is the smallest prime in the range [floor(n/2)+1,n]

        Required Args
        -------------

        n : int
            dimensionality of the search space
        
        Returns
        -------
        The shuffle stride int
        """
        def _isprime(x):
            if x <= 1:
                return False
            elif x <= 3:
                return True
            elif (x % 2 == 0) or (x % 3 == 0):
                return False
            i = 5
            while i*i <= x:
                if (x % i == 0) or (x % (i+2) == 0):
                    return False
                i += 6
            return True
        
        for p in range(n//2 + 1, n):
            if _isprime(p):
                return p
        raise ValueError("No prime found for n={}".format(n))

    @staticmethod
    def getSampStride(n):
        """
        Calculate the mesh sampling stride.
        sample stride is the first odd value >= log2(n) or 1 if n < 10.

        Required Args
        -------------

        n : int
            dimensionality of the search space

        Returns
        -------
        The sample stride int
        """
        if n < 10:
            return 1
        return int(2*torch.floor(torch.log2(torch.tensor(n))/2)+1)

    @staticmethod
    def validateMesh(M, tol=1e-5):
        """
        unit test function to check that:
          1. the mesh columns are unit vectors
          2. every dihedral angle is equal to arccos(-1/n)

        Required Args
        -------------

        M : Tensor
            a mesh to evaluate for validity 

        Optional Args
        -------------

        tol : float (default=1e-5)
            tolerance for floating point assertions

        Returns
        -------
        True if the mesh columns come from a unit regular simplex,
        else False
        """
        if any(abs(torch.linalg.norm(M,dim=0)-1) > tol):
            return False
        targetAngle = torch.acos(torch.tensor(-1/M.shape[0]))
        for c in range(M.shape[1]-1):
            dihedralAngles = torch.acos(torch.sum(
                torch.unsqueeze(M[:,c],1)*M[:,c+1:M.shape[1]],0))
            if any(abs(dihedralAngles - targetAngle) > tol):
                return False
        return True
