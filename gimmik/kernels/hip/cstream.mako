# -*- coding: utf-8 -*-

<% ksplit = 2 if m < 36 else 1 %>

% if dtype == 'float2':
inline __device__ float2 operator*(float a, float2 b)
{ return make_float2(a*b.x, a*b.y); }

inline __device__ float2 operator+(float2 a, float2 b)
{ return make_float2(a.x + b.x, a.y + b.y); }

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
% for j, jx in enumerate(A):
  % if beta == 0:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=ksplit)};
  % elif beta == 1:
        c[i + ${j}*ldc] += ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=ksplit)};
  % else:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=ksplit)}
                        + ${beta}*c[i + ${j}*ldc];
  % endif
% endfor
    }
}
