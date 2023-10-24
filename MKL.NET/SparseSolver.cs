// Copyright 2023 Victor Ceballos
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace MKLNET;

using System.Security;
using System.Runtime.InteropServices;

[SuppressUnmanagedCodeSecurity]
public static partial class SparseSolver
{
    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "pardisoinit")]
    internal static extern unsafe void _pardisoinit([In, Out] System.IntPtr[] handle, ref int mtype, [In, Out] int[] iparm);

    public static void pardisoinit(System.IntPtr[] handle, ref int mtype, int[] iparm)
    {
        _pardisoinit(handle, ref mtype, iparm);
    }

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "pardiso")]
    internal static extern unsafe void _pardiso([In, Out] System.IntPtr[] handle, ref int maxfct, ref int mnum, ref int mtype,
        ref int phase, ref int n, [In] double[] a, [In] int[] ia, [In] int[] ja, [In] int[] perm, ref int nrhs,
        [In, Out] int[] iparm, ref int msglvl, [In, Out] double[] b, [Out] double[] x, out int error);

    public static void pardiso(System.IntPtr[] handle, ref int maxfct, ref int mnum, ref int mtype, ref int phase, ref int n, double[] a, int[] ia, int[] ja, int[] perm, ref int nrhs, int[] iparm, ref int msglvl, double[] b, double[] x, out int error)
    {
        _pardiso(handle, ref maxfct, ref mnum, ref mtype, ref phase, ref n, a, ia, ja, perm, ref nrhs, iparm, ref msglvl, b, x, out error);
    }
}
