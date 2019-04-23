﻿// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ApplyUtils.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Struct TensorInfo
    /// </summary>
    public struct TensorInfo
    {
        /// <summary>
        /// The sizes
        /// </summary>
        public long[] sizes;
        /// <summary>
        /// The strides
        /// </summary>
        public long[] strides;
        /// <summary>
        /// The buffer
        /// </summary>
        public CUdeviceptr buffer;
    }

    /// <summary>
    /// Class ApplyUtils.
    /// </summary>
    public static class ApplyUtils
    {
        /// <summary>
        /// The apply threads per block
        /// </summary>
        public const int ApplyThreadsPerBlock = 32 * 16;


        /// <summary>
        /// Gets the apply block.
        /// </summary>
        /// <returns>dim3.</returns>
        public static dim3 GetApplyBlock()
        {
            return new dim3(ApplyThreadsPerBlock);
        }

        // returns Ceil(x / y)
        /// <summary>
        /// Ceils the div.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns>System.Int64.</returns>
        public static long CeilDiv(long x, long y)
        {
            return (x + y - 1) / y;
        }

        /// <summary>
        /// Gets the apply grid.
        /// </summary>
        /// <param name="deviceInfo">The device information.</param>
        /// <param name="totalElements">The total elements.</param>
        /// <returns>dim3.</returns>
        public static dim3 GetApplyGrid(CudaDeviceProperties deviceInfo, long totalElements)
        {
            var smCount = deviceInfo.MultiProcessorCount;

            // Rationale for grid size - from cuTorch source code:
            // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
            // which seems to be a good sweetspot for latency hiding
            var maxSize = 4 * smCount;
            var targetSize = CeilDiv(totalElements, ApplyThreadsPerBlock);
            return new dim3((uint)Math.Min(targetSize, maxSize));
        }



        /// <summary>
        /// Determines whether this instance [can use32 bit index math] the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns><c>true</c> if this instance [can use32 bit index math] the specified tensor; otherwise, <c>false</c>.</returns>
        public static bool CanUse32BitIndexMath(NDArray tensor)
        {
            var elements = tensor.ElementCount();
            if (elements >= uint.MaxValue)
            {
                return false;
            }

            long offset = 0;
            long linearId = elements - 1;

            for (int i = tensor.DimensionCount - 1; i >= 0; --i)
            {
                var curDimIndex = linearId % tensor.Shape[i];
                var curDimOffset = curDimIndex * tensor.Strides[i];
                offset += curDimOffset;
                linearId /= tensor.Shape[i];
            }

            if (offset >= uint.MaxValue)
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Struct SizeAndStride
        /// Implements the <see cref="System.IComparable" />
        /// </summary>
        /// <seealso cref="System.IComparable" />
        private struct SizeAndStride : IComparable
        {
            /// <summary>
            /// The size
            /// </summary>
            public long size; public long stride;

            /// <summary>
            /// Compares to.
            /// </summary>
            /// <param name="obj">The object.</param>
            /// <returns>System.Int32.</returns>
            /// <exception cref="InvalidOperationException"></exception>
            public int CompareTo(object obj)
            {
                if (!(obj is SizeAndStride)) throw new InvalidOperationException();
                var o = (SizeAndStride)obj;
                return -stride.CompareTo(o.stride); // negated because we require descending order
            }
        }

        /// <summary>
        /// Determines whether [has overlapping indices] [the specified tensor].
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns><c>true</c> if [has overlapping indices] [the specified tensor]; otherwise, <c>false</c>.</returns>
        public static bool HasOverlappingIndices(NDArray tensor)
        {
            // In this function, we don't care about permutations of the
            // size/stride arrays (transpositions).
            // We order the size/stride arrays by stride, skipping dimensions of
            // size 1. Strides of dimensions of size 1 don't matter, since there
            // is only one addressing point in them.
            // In this reordered view, the tensor is contiguous if
            // stride[dim] == size[dim + 1] * stride[dim + 1] for all `dim`.
            // The tensor has holes if
            // stride[dim] > size[dim + 1] * stride[dim + 1] for one or more
            // `dim`.
            // The tensor has overlaps if
            // stride[dim] < size[dim + 1] * stride[dim + 1] for one or more
            // `dim`, or the innermost stride is 0.

            // Extract size/stride arrays; only consider size >1 dims.
            var info = Enumerable.Range(0, tensor.DimensionCount)
                .Select(x => new SizeAndStride() { size = tensor.Shape[x], stride = tensor.Strides[x] })
                .Where(x => x.size > 1)
                .ToArray();

            if (info.Length == 0)
            {
                return false; // no overlap
            }

            // Descending order (innermost dimension in sorted view is at last index)
            Array.Sort(info);

            // Base case: innermost dimension must have stride >= 1
            if (info[info.Length - 1].stride < 1)
            {
                return true;
            }

            // Subsequent dimensions, if any
            for (int i = info.Length - 2; i >= 0; --i)
            {
                if (info[i].stride < info[i + 1].size * info[i + 1].stride)
                {
                    // There are overlaps
                    return true;
                }
            }

            // Tensor has holes or is contiguous
            return false;
        }



        /// <summary>
        /// Gets the innermost non1 dim.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <param name="excludeDim">The exclude dim.</param>
        /// <returns>System.Int32.</returns>
        private static int GetInnermostNon1Dim(long[] sizes, int excludeDim)
        {
            for (int i = sizes.Length - 1; i >= 0; --i)
            {
                if (i == excludeDim)
                {
                    return i;
                }

                if (sizes[i] != 1)
                {
                    return i;
                }
            }

            return -1;
        }

        //TODO this is not actually used at the moment. It probably should be.
        // excludeDim may be -1 to not exclude any dimension
        /// <summary>
        /// Collapses the dims.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="excludeDim">The exclude dim.</param>
        /// <param name="info">The information.</param>
        /// <returns>System.Int32.</returns>
        /// <exception cref="ArgumentException">excludeDim must equal -1 if all dims are of size 1 - excludeDim</exception>
        public static int CollapseDims(NDArray tensor, int excludeDim, out TensorInfo info)
        {
            info.buffer = CudaHelpers.GetBufferStart(tensor);
            var firstNonOneDim = GetInnermostNon1Dim(tensor.Shape, excludeDim);

            // If all dims are size 1 (ie. tensor contains 1 element)
            if (firstNonOneDim == -1)
            {
                if (excludeDim != -1) throw new ArgumentException("excludeDim must equal -1 if all dims are of size 1", "excludeDim");

                info.sizes = new long[] { 1 };
                info.strides = new long[] { 1 };
                return 0;
            }


            // Count the number of successive dimensions that can be collapsed, from
            // innermost to outermost.
            int numCollapsed = 0;

            // Skip the leading size 1 dims
            numCollapsed += tensor.DimensionCount - 1 - firstNonOneDim;

            // We perform one pass through to determine how many dimensions we
            // can collapse, before calculating the actual size of the collapsed
            // dimensions.
            // size/strideInner are the size/strides of the previous inner
            // non-collapsible dim we encounter.
            var sizeInner = tensor.Shape[firstNonOneDim];
            var strideInner = tensor.Strides[firstNonOneDim];

            for (int i = firstNonOneDim - 1; i >= 0; --i)
            {
                var sizeOuter = tensor.Shape[i];
                var strideOuter = tensor.Strides[i];

                // Don't collapse this dimension if we want to exclude it from
                // collapsing.
                // Since this code is attempting to collapse a subsequent
                // dimension (i) with the preceding dimension (i + 1), we can only
                // perform collapsing if the preceding dimension can be collapsed
                // (i.e., not excludeDim)
                if ((excludeDim != i) && (excludeDim != i + 1))
                {
                    // The next outermost dimension can be skipped if size 1
                    if (sizeOuter == 1)
                    {
                        ++numCollapsed;
                        continue;
                    }

                    // If the next outermost dimension is contiguous with the
                    // previous non-collapsed one, collapse it
                    if (strideOuter == strideInner * sizeInner)
                    {
                        ++numCollapsed;

                        // This is the run of collapsed dimensions' size
                        sizeInner = sizeInner * sizeOuter;
                        continue;
                    }
                }

                // Otherwise, this new outer dimension at `i` cannot be collapsed
                // because it is excluded from collapsing, or it is not contiguous
                // with the previous inner dimension.
                sizeInner = sizeOuter;
                strideInner = strideOuter;
            }

            // This will be our new size/stride and dimension.
            var newSizes = new long[TSCudaContext.MaxDims];
            var newStrides = new long[TSCudaContext.MaxDims];

            int newDims = tensor.DimensionCount - numCollapsed;

            // We return the index of the excluded dimension that is excluded
            // from being collapsed here.
            int returnDim = -1;

            // We perform a second pass through the dimensions to actually
            // calculate the size of the collapsed dimensions.
            int collapsedIndex = tensor.DimensionCount - numCollapsed - 1;
            newSizes[collapsedIndex] = tensor.Shape[firstNonOneDim];
            newStrides[collapsedIndex] = tensor.Strides[firstNonOneDim];

            if (firstNonOneDim == excludeDim)
            {
                returnDim = collapsedIndex;
            }

            for (int i = firstNonOneDim - 1; i >= 0; --i)
            {
                var sizeOuter = tensor.Shape[i];
                var strideOuter = tensor.Strides[i];

                if ((excludeDim != i) && (excludeDim != i + 1))
                {
                    if (sizeOuter == 1)
                    {
                        // skip
                        continue;
                    }

                    if (strideOuter == newSizes[collapsedIndex] * newStrides[collapsedIndex])
                    {
                        // collapse
                        newSizes[collapsedIndex] *= sizeOuter;
                        continue;
                    }
                }

                // Otherwise, strides don't match, or dim `i` is excluded from
                // collapsing.
                --collapsedIndex;
                //assert(collapsedIndex >= 0);
                //assert(collapsedIndex < newDims);
                newSizes[collapsedIndex] = sizeOuter;
                newStrides[collapsedIndex] = strideOuter;

                if (excludeDim == i)
                {
                    returnDim = collapsedIndex;
                }

            }

            info.sizes = newSizes.Take(newDims).ToArray();
            info.strides = newStrides.Take(newDims).ToArray();
            return returnDim;
        }
    }
}
