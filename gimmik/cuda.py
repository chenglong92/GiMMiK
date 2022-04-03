# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class CUDAMatMul(MatMul):
    platform = 'cuda'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    def _kernel_generators(self, dtype, dsize):
        # B loading, C streaming kernel
        yield ('cstream', {}, {})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 16, 64
        args = {'msplit': ms, 'blockx': blkx, 'bsz': bsz}
        meta = {'block': (blkx, ms, 1), 'shared': 2*blkx*bsz*dsize}
        yield ('bstream-msplit', args, meta)

        # At single precision also consider vectorized kernels
        if (dtype == 'float' and not self.has_zero_rows and
            self.aligne is not None and self.aligne % 2 == 0):
            # Vector B loading, C streaming kernel
            args = {'dtype': 'float2', 'width': 2}
            meta = {'width': 2}
            yield ('cstream', args, meta)

            # Vector four-way m-split B streaming, C accumulation kernel
            ms, bsz, blkx = 4, 16, 64
            args = {'dtype': 'float2', 'width': 2, 'msplit': ms,
                    'blockx': blkx, 'bsz': bsz}
            meta = {'block': (blkx, ms, 1), 'width': 2,
                    'shared': 2*blkx*bsz*2*dsize}
            yield ('bstream-msplit', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
