# -*- coding: utf-8 -*-

__global__ __launch_bounds__(128) void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    ${dtype} dotp;

    if (i < n)
    {
    % for j, jx in enumerate(mat):
        dotp = ${' + '.join(f'{kx}*b[i + {k}*ldb]'
                            for k, kx in enumerate(jx) if kx != 0) or 0};
    % if beta == 0:
        c[i + ${j}*ldc] = dotp;
    % elif beta == 1:
        c[i + ${j}*ldc] += dotp;
    % else:
        c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
    % endif
    % endfor
    }
}