# -*- coding: utf-8 -*-

import itertools as it
import pkgutil
import re

from mako.template import Template
import numpy as np


def _dot(bfn, row, maxsplit=1):
    nzixs, = np.nonzero(row)

    if not nzixs.size:
        return '0'

    nsplit = max(min(maxsplit, nzixs.size // 3), 1)
    snzixs = np.array_split(nzixs, nsplit)

    frags = [' + '.join(f'{row[i]}*{bfn(i)}' for i in ix) for ix in snzixs]
    return ' + '.join(f'({f})' for f in frags)


def _splitm(mat, maxsplit=1):
    smat = [mat[i::maxsplit] for i in range(maxsplit)]
    sidx = [list(range(i, len(mat), maxsplit)) for i in range(maxsplit)]

    return smat, sidx


def _chunk(l, chunksz):
    l, n = iter(l), len(l)
    nchunks = -(-n // chunksz)

    return [list(it.islice(l, chunksz)) for i in range(nchunks)]


class MatMul:
    platform = None

    def __init__(self, A, beta=0.0, aligne=None, n=None, ldb=None, ldc=None):
        self.A = A
        self.beta = beta
        self.aligne = aligne

        if n is None and ldb is None and ldc is None:
            self.n = self.ldb = self.ldc = None
        elif n is not None and ldb is not None and ldc is not None:
            if aligne is not None and (ldb % aligne or ldc % aligne):
                raise ValueError('ldb/ldc not compatible with aligne')

            self.n, self.ldb, self.ldc = n, ldb, ldc
        else:
            raise ValueError('Must provide all of (n, ldb, ldc) or none')

        # Extract the shape of A
        self.m, self.k = m, k = A.shape

        # Determine the index of the first and last non-zero in each row of A
        self.afix = (A != 0).argmax(axis=1)
        self.alix = k - 1 - (A != 0)[:, ::-1].argmax(axis=1)

        # Mark rows of A which are all zero
        self.afix = np.where(np.any(A != 0, axis=1), self.afix, -1)
        self.alix = np.where(np.any(A != 0, axis=1), self.alix, -1)
        self.has_zero_rows = np.any(self.afix == -1)

        # Determine which entries of B partake in the multiplication
        self.bix = np.nonzero(np.any(A != 0, axis=0))[0]
        self.bix = {kx: k for k, kx in enumerate(self.bix)}

    def kernels(self, dtype, kname='gimmik_mm', **kwargs):
        basemeta = self.basemeta

        # Process the data type
        dtype = np.dtype(dtype).type
        if dtype == np.float32:
            dtype, dsize = 'float', 4
        elif dtype == np.float64:
            dtype, dsize = 'double', 8
        else:
            raise ValueError('Invalid floating point data type')

        # Common template arguments
        baseargs = {
            'dtype': dtype, 'kname': kname,
            'A': self.A, 'beta': self.beta, 'width': 1,
            'm': self.m, 'n': self.n, 'k': self.k,
            'ldb': self.ldb, 'ldc': self.ldc,
            'afix': self.afix, 'alix': self.alix, 'bix': self.bix,
            'dot': _dot, 'splitm': _splitm, 'chunk': _chunk
        }

        # Incrementally generate and render the kernels
        for name, exargs, exmeta in self._kernel_generators(dtype, dsize):
            # Merge in the base arguments and metadata
            args = baseargs | exargs
            meta = basemeta | exmeta

            # Render the kernel template
            src = self._render_kernel(dtype, name, args)

            # Post-process the metadata
            meta['tplname'] = name
            self._process_meta(meta)

            yield (src, meta)

    def _process_meta(self, meta):
        pass

    def _render_kernel(self, dtype, tplname, tplargs):
        platform = self.platform

        # Load and render the template
        tpl = pkgutil.get_data(__name__, f'kernels/{platform}/{tplname}.mako')
        src = Template(tpl).render(**tplargs)

        # At single precision suffix all floating point constants by 'f'
        if dtype == 'float':
            src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                         r'\g<0>f', src)

        # Cleanup
        src = re.sub(r'\n\n+', r'\n\n', src.strip()) + '\n'
        src = re.sub(r'\w+$', '', src)
        return src
