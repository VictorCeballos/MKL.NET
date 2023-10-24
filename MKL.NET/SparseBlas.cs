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
using System.Runtime.CompilerServices;

[SuppressUnmanagedCodeSecurity]
public static partial class SparseBlas
{
    public unsafe static class Unsafe
    {
        // Matrix Manipulation Routines

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_s_create_csr")]
        public static extern SparseStatus create_csr(out System.IntPtr A, SparseIndexBase indexing, int rows, int cols, [In] int* rows_start, [In] int* rows_end, [In] int* col_indx, [In] float* values);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_d_create_csr")]
        public static extern SparseStatus create_csr(out System.IntPtr A, SparseIndexBase indexing, int rows, int cols, [In] int* rows_start, [In] int* rows_end, [In] int* col_indx, [In] double* values);

        // Inspector-Executor Sparse BLAS Execution Routines

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_s_mv")]
        public static extern SparseStatus mv(SparseOperation operation, float alpha, System.IntPtr A, SparseBlas.matrix_descr descr, [In] float* x, float beta, [In, Out] float* y);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_d_mv")]
        public static extern SparseStatus mv(SparseOperation operation, double alpha, System.IntPtr A, SparseBlas.matrix_descr descr, [In] double* x, double beta, [In, Out] double* y);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_s_mm")]
        public static extern SparseStatus mm(SparseOperation operation, float alpha, System.IntPtr A, SparseBlas.matrix_descr descr, SparseLayout layout, [In] float* x, int columns, int ldx, float beta, [In, Out] float* y, int ldy);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_d_mm")]
        public static extern SparseStatus mm(SparseOperation operation, double alpha, System.IntPtr A, SparseBlas.matrix_descr descr, SparseLayout layout, [In] double* x, int columns, int ldx, double beta, [In, Out] double* y, int ldy);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_s_trsv")]
        public static extern SparseStatus trsv(SparseOperation operation, float alpha, System.IntPtr A, SparseBlas.matrix_descr descr, [In] float* x, [Out] float* y);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_d_trsv")]
        public static extern SparseStatus trsv(SparseOperation operation, double alpha, System.IntPtr A, SparseBlas.matrix_descr descr, [In] double* x, [Out] double* y);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_s_trsm")]
        public static extern SparseStatus trsm(SparseOperation operation, float alpha, System.IntPtr A, SparseBlas.matrix_descr descr, SparseLayout layout, [In] float* x, int columns, int ldx, [In, Out] float* y, int ldy);

        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_d_trsm")]
        public static extern SparseStatus trsm(SparseOperation operation, double alpha, System.IntPtr A, SparseBlas.matrix_descr descr, SparseLayout layout, [In] double* x, int columns, int ldx, [In, Out] double* y, int ldy);
    }

    // Matrix Manipulation Routines

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern unsafe int mkl_sparse_s_export_csr([In] System.IntPtr source, [In, Out] ref int indexing, [In, Out] ref int rows, [In, Out] ref int cols, [In, Out] ref System.IntPtr rows_start, [In, Out] ref System.IntPtr rows_end, [In, Out] ref System.IntPtr col_indx, [In, Out] ref System.IntPtr values);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static SparseStatus export_csr(System.IntPtr source, out SparseIndexBase indexing, out int rows, out int cols, out int[] rows_start, out int[] rows_end, out int[] col_indx, out float[] values)
    {
        int ind = 0;
        int m = 0;
        int k = 0;
        System.IntPtr rows_start_ptr = System.IntPtr.Zero;
        System.IntPtr rows_end_ptr = System.IntPtr.Zero;
        System.IntPtr col_indx_ptr = System.IntPtr.Zero;
        System.IntPtr values_ptr = System.IntPtr.Zero;
        SparseStatus status = (SparseStatus)mkl_sparse_s_export_csr(source, ref ind, ref m, ref k, ref rows_start_ptr, ref rows_end_ptr, ref col_indx_ptr, ref values_ptr);

        indexing = (SparseIndexBase)ind;
        rows = m;
        cols = k;
        rows_start = new int[rows];
        Marshal.Copy(rows_start_ptr, rows_start, 0, rows);
        rows_end = new int[rows];
        Marshal.Copy(rows_end_ptr, rows_end, 0, rows);
        int nnz = rows_end[rows - 1] - ind;
        col_indx = new int[nnz];
        Marshal.Copy(col_indx_ptr, col_indx, 0, nnz);
        values = new float[nnz];
        Marshal.Copy(values_ptr, values, 0, nnz);

        return status;
    }

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern unsafe int mkl_sparse_d_export_csr([In] System.IntPtr source, [In, Out] ref int indexing, [In, Out] ref int rows, [In, Out] ref int cols, [In, Out] ref System.IntPtr rows_start, [In, Out] ref System.IntPtr rows_end, [In, Out] ref System.IntPtr col_indx, [In, Out] ref System.IntPtr values);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static SparseStatus export_csr(System.IntPtr source, out SparseIndexBase indexing, out int rows, out int cols, out int[] rows_start, out int[] rows_end, out int[] col_indx, out double[] values)
    {
        int ind = 0;
        int m = 0;
        int k = 0;
        System.IntPtr rows_start_ptr = System.IntPtr.Zero;
        System.IntPtr rows_end_ptr = System.IntPtr.Zero;
        System.IntPtr col_indx_ptr = System.IntPtr.Zero;
        System.IntPtr values_ptr = System.IntPtr.Zero;
        SparseStatus status = (SparseStatus)mkl_sparse_d_export_csr(source, ref ind, ref m, ref k, ref rows_start_ptr, ref rows_end_ptr, ref col_indx_ptr, ref values_ptr);

        indexing = (SparseIndexBase)ind;
        rows = m;
        cols = k;
        rows_start = new int[rows];
        Marshal.Copy(rows_start_ptr, rows_start, 0, rows);
        rows_end = new int[rows];
        Marshal.Copy(rows_end_ptr, rows_end, 0, rows);
        int nnz = rows_end[rows - 1] - ind;
        col_indx = new int[nnz];
        Marshal.Copy(col_indx_ptr, col_indx, 0, nnz);
        values = new double[nnz];
        Marshal.Copy(values_ptr, values, 0, nnz);

        return status;
    }

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_order")]
    public static extern SparseStatus order(System.IntPtr csrA);

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_destroy")]
    public static extern SparseStatus destroy(System.IntPtr A);

    // Inspector-Executor Sparse BLAS Analysis Routines

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_set_mv_hint")]
    public static extern SparseStatus set_mv_hint(System.IntPtr A, SparseOperation operation, SparseBlas.matrix_descr descr, int expected_calls);

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_set_sv_hint")]
    public static extern SparseStatus set_sv_hint(System.IntPtr A, SparseOperation operation, SparseBlas.matrix_descr descr, int expected_calls);

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_set_mm_hint")]
    public static extern SparseStatus set_mm_hint(System.IntPtr A, SparseOperation operation, SparseBlas.matrix_descr descr, SparseLayout layout, int dense_matrix_size, int expected_calls);

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_optimize")]
    public static extern SparseStatus optimize(System.IntPtr A);

    // Inspector-Executor Sparse BLAS Execution Routines

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_spmm")]
    public static extern SparseStatus spmm(SparseOperation operation, System.IntPtr A, System.IntPtr B, out System.IntPtr C);

    [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, EntryPoint = "mkl_sparse_syrk")]
    public static extern SparseStatus syrk(SparseOperation operation, System.IntPtr A, out System.IntPtr C);

    /* descriptor of main sparse matrix properties */
    [StructLayout(LayoutKind.Explicit, Size = 12)]
    public struct matrix_descr
    {
        [FieldOffset(0)]
        public SparseMatrixType type;           /* matrix type: general, diagonal or triangular / symmetric / hermitian */
        [FieldOffset(4)]
        public SparseFillMode mode;             /* upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case) */
        [FieldOffset(8)]
        public SparseDiagType diag;             /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
    }
}
