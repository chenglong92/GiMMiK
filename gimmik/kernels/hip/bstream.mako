# -*- coding: utf-8 -*-

% if dtype == 'float2':
inline __device__ float2 operator*(float a, float2 b)
{ return make_float2(a*b.x, a*b.y); }

inline __device__ void operator+=(float2 &a, float2 b)
{ a.x += b.x; a.y += b.y; }
% endif

__global__ __launch_bounds__(128) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const int ldb = ${ldb // width};
    const int ldc = ${ldc // width};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < n)
    {
        ${dtype} bv, csub[${m}];

## Iterare through the used rows of B
% for kx in bix:
        bv = b[i + ${kx}*ldb];
  % for j, jx in enumerate(A[:, kx]):
    % if jx != 0 and kx == afix[j]:
        csub[${j}] = ${jx}*bv;
    % elif jx != 0:
        csub[${j}] += ${jx}*bv;
    % endif
    ##
    % if kx == alix[j] and beta == 0:
        c[i + ${j}*ldc] = csub[${j}];
    % elif kx == alix[j] and beta == 1:
        c[i + ${j}*ldc] += csub[${j}];
    % elif kx == alix[j]:
        c[i + ${j}*ldc] = csub[${j}] + ${beta}*c[i + ${j}*ldc];
    % endif
  % endfor
% endfor

## Handle rows of A which are all zero
% for j, jx in enumerate(afix):
  % if jx == -1 and beta == 0:
        c[i + ${j}*ldc] = 0;
  % elif jx == -1 and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
  % endif
% endfor
    }
}
