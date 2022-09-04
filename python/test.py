#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert
All Rights Reserved

SpartaPlex Academic License

test.py
------------------------------------------------------------------------
"""

import torch
import time

from spartaplex import SpartaPlex


def main():

    # read random spheroid center
    with open("randcenter.txt" , 'r') as f:
        center = [float(x.strip()) for x in f.readlines()]
    center = torch.tensor(center).unsqueeze(1)

    n = center.shape[0]

    # spheroid function handle
    f = lambda x: torch.sum(torch.square(x-center), dim=0)

    print("Optimizing {}-D spheroid...".format(n), end="", flush=True)
    start = time.time()
    sp = SpartaPlex(n)
    minvec, minval = sp.optimize(f)
    print("finished in {:0.2f} seconds".format(time.time()-start))

    print("minval = {:0.5g}".format(minval))


if __name__ == "__main__":
    main()
