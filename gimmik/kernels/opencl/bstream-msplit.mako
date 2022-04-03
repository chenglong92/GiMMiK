# -*- coding: utf-8 -*-

<%
ms, mx = splitm(A, msplit)
bchunks = chunk(bix, bsz)
%>

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
    int lx = get_local_id(0), ly = get_local_id(1);

    ${dtype} bv, csub[${-(-m // msplit)}];
    __local ${dtype} bsub[2][${bsz}][${blockx}];

## Iterate over each chunk of C
% for cid, mcs, mcx in zip(range(msplit), ms, mx):
    if (ly == ${cid})
    {
  ## Iterate over each chunk of B
  % for bb in range(len(bchunks)):
    ## Fill the initial shared memory block
    % if loop.first:
        if (i < n)
        {
      % for kx in bchunks[bb]:
        % if loop.index % msplit == cid:
            bsub[0][${loop.index}][lx] = b[i + ${kx}*ldb];
        % endif
      % endfor
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    % endif
        if (i < n)
        {
    ## Start filling the next shared memory block
    % if not loop.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
            bsub[${(bb + 1) % 2}][${loop.index}][lx] = b[i + ${kx}*ldb];
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
            bv = bsub[${bb % 2}][${loop.index}][lx];
      % for j, jx in enumerate(mcs[:, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
            csub[${j}] = ${jx}*bv;
        % elif jx != 0:
            csub[${j}] += ${jx}*bv;
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]] and beta == 0:
            c[i + ${mcx[j]}*ldc] = csub[${j}];
        % elif kx == alix[mcx[j]] and beta == 1:
            c[i + ${mcx[j]}*ldc] += csub[${j}];
        % elif kx == alix[mcx[j]]:
            c[i + ${mcx[j]}*ldc] = csub[${j}] + ${beta}*c[i + ${mcx[j]}*ldc];
        % endif
      % endfor
    % endfor
        }
        barrier(CLK_LOCAL_MEM_FENCE);
  % endfor

  ## Handle rows of A which are all zero
  % for j, jx in enumerate(afix):
    % if jx == -1 and j % msplit == cid and beta == 0:
        c[i + ${j}*ldc] = 0;
    % elif jx == -1 and j % msplit == cid and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
    % endif
  % endfor
    }
% endfor
}
