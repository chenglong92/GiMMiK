# -*- coding: utf-8 -*-

% if dtype == 'double':
#if __OPENCL_VERSION__ < 120
# pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
% endif

__kernel void
% if n is None:
${kname}(int n,
         __global const ${dtype}* restrict b, int ldb,
         __global ${dtype}* restrict c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(__global const ${dtype}* restrict b, __global ${dtype}* restrict c)
{
    const int n = ${-(-n // width)};
    const int ldb = ${ldb // width};
    const int ldc = ${ldc // width};
% endif
    int i = get_global_id(0);

    if (i < n)
    {
% for j, jx in enumerate(A):
  % if beta == 0:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=3)};
  % elif beta == 1:
        c[i + ${j}*ldc] += ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=3)};
  % else:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=3)}
                        + ${beta}*c[i + ${j}*ldc];
  % endif
% endfor
    }
}
