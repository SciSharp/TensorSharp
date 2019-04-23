using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;

namespace TensorSharpTests
{
    [TestClass]
    public class FillTests
    {
        [TestMethod]
        public void FillByte()
        {
            var allocator = new CpuAllocator();
            var a = new NDArray(allocator, DType.UInt8, 1);

            var value = 97f;
            Ops.Fill(a, value);

            Assert.AreEqual(value, a.GetElementAsFloat(0));
        }
    }
}
