# -*- coding: utf-8 -*-

from gimmik._version import __version__
from gimmik.c import CMatMul,
from gimmik.copenmp import COpenMPMatMul
from gimmik.cuda import CUDAMatMul
from gimmik.ispc import ISPCMatMul
from gimmik.hip import HIPMatMul
from gimmik.opencl import OpenCLMatMul

        #if platform in {'c', 'c-omp', 'ispc'}:
            #generators.append((platform, baseargs, {}))
        #elif platform == 'hip':
            #basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

            ## B loading, C streaming kernel
            #generators.append(('hip-cstream', baseargs, basemeta))

            ## B streaming, C accumulation kernel
            #generators.append(('hip-bstream', baseargs, basemeta))

            ## If n is known also fill out the grid size
            #if n is not None:
                #for tplname, tplargs, meta in generators:
                    #meta['grid'] = (-(-n // meta['block'][0]), 1, 1)


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm',
                n=None, ldb=None, ldc=None):
    import warnings

    warnings.warn('generate_mm is deprecated, use MatMul', DeprecationWarning)

    platmap = {
        'c': CMatMul,
        'c-omp': COpenMPMatMul,
        'cuda': CUDAMatMul,
        'ispc': ISPCMatMul,
        'hip': HIPMatMul,
        'opencl': OpenCLMatMul
    }

    mm = platmap[platform](alpha*mat, beta, None, n, ldb, ldc)
    return next(mm.kernels(dtype, kname=funcn))[0]
