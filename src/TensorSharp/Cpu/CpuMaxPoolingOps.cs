﻿// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuMaxPoolingOps.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuMaxPoolingOps.
    /// </summary>
    public static class CpuMaxPoolingOps
    {
        /// <summary>
        /// Outputs the size.
        /// </summary>
        /// <param name="inputSizes">The input sizes.</param>
        /// <param name="ceilMode">if set to <c>true</c> [ceil mode].</param>
        /// <param name="cd">The cd.</param>
        /// <returns>System.Int64[].</returns>
        public static long[] OutputSize(long[] inputSizes, bool ceilMode, ConvolutionDesc2d cd)
        {
            int dimw = 3;
            int dimh = 2;

            var iwidth = inputSizes[dimw];
            var iheight = inputSizes[dimh];

            long oheight, owidth;
            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            return new long[] { inputSizes[0], inputSizes[1], oheight, owidth };
        }


        /// <summary>
        /// Spatials the maximum pooling forward.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="indices">The indices.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="ceilMode">if set to <c>true</c> [ceil mode].</param>
        /// <exception cref="ArgumentException">input must be a 4D tensor</exception>
        /// <exception cref="InvalidOperationException">
        /// input image is smaller than kernel size
        /// or
        /// pad should be smaller than half of the kernel size
        /// </exception>
        public static void SpatialMaxPoolingForward(NDArray input, NDArray output, NDArray indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            if (input.DimensionCount != 4) throw new ArgumentException("input must be a 4D tensor");

            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            if (input.Shape[dimw] < cd.kW - cd.padW || input.Shape[dimh] < cd.kH - cd.padH)
                throw new InvalidOperationException("input image is smaller than kernel size");

            if (cd.padW > cd.kW / 2 || cd.padH > cd.kH / 2)
                throw new InvalidOperationException("pad should be smaller than half of the kernel size");

            var nbatch = input.Shape[0];
            var nslices = input.Shape[dimc];
            var iheight = input.Shape[dimh];
            var iwidth = input.Shape[dimw];

            long owidth;
            long oheight;

            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            if (cd.padW != 0 || cd.padH != 0)
            {
                // ensure that the last pooling starts inside the image
                if ((oheight - 1) * cd.dH >= iheight + cd.padH)
                    --oheight;
                if ((owidth - 1) * cd.dW >= iwidth + cd.padW)
                    --owidth;
            }

            using (var inputContig = Ops.AsContiguous(input))
            {
                for (int i = 0; i < nbatch; ++i)
                {
                    using (var input_i = inputContig.Select(0, i))
                    using (var output_i = output.Select(0, i))
                    using (var indices_i = indices.Select(0, i))
                    {
                        IntPtr input_iPtr, output_iPtr, indices_iPtr;
                        using (NativeWrapper.BuildTensorRefPtr(input_i, out input_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(output_i, out output_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(indices_i, out indices_iPtr))
                        {
                            CpuOpsNative.TS_SpatialMaxPooling_updateOutput_frame(input_iPtr, output_iPtr, indices_iPtr,
                                nslices, iwidth, iheight,
                                owidth, oheight,
                                cd.kW, cd.kH, cd.dW, cd.dH, cd.padW, cd.padH);
                        }
                    }
                }
            }

        }


        /// <summary>
        /// Spatials the maximum pooling backward.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradInput">The grad input.</param>
        /// <param name="indices">The indices.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="ceilMode">if set to <c>true</c> [ceil mode].</param>
        public static void SpatialMaxPoolingBackward(NDArray input, NDArray gradOutput, NDArray gradInput, NDArray indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            var nbatch = input.Shape[0];
            var nslices = input.Shape[dimc];
            var iheight = input.Shape[dimh];
            var iwidth = input.Shape[dimw];
            var owidth = gradOutput.Shape[dimw];
            var oheight = gradOutput.Shape[dimh];

            Ops.Fill(gradInput, 0);


            using (var gradOutputContig = Ops.AsContiguous(gradOutput))
            {
                for (int i = 0; i < nbatch; ++i)
                {
                    using (var gradInput_i = gradInput.Select(0, i))
                    using (var gradOutput_i = gradOutputContig.Select(0, i))
                    using (var indices_i = indices.Select(0, i))
                    {
                        IntPtr gradInput_iPtr, gradOutput_iPtr, indices_iPtr;
                        using (NativeWrapper.BuildTensorRefPtr(gradInput_i, out gradInput_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(gradOutput_i, out gradOutput_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(indices_i, out indices_iPtr))
                        {
                            CpuOpsNative.TS_SpatialMaxPooling_updateGradInput_frame(gradInput_iPtr, gradOutput_iPtr, indices_iPtr,
                                nslices, iwidth, iheight,
                                owidth, oheight,
                                cd.dW, cd.dH);
                        }
                    }
                }
            }
        }
    }
}
