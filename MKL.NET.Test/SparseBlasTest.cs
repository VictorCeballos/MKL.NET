namespace MKL.NET.Test;

using System;
using System.Linq;
using Xunit;
using MKLNET;


public class SparseBlasTest
{
    /*******************************************************************************
    *   Content : Intel(R) oneAPI Math Kernel Library (oneMKL) IE Sparse BLAS C
    *             example for CSR format
    *
    ********************************************************************************
    *
    * Example program for using Intel oneMKL Inspector-Executor Sparse BLAS routines
    * for matrices represented in the compressed sparse row (CSR) sparse storage format.
    *
    * The following Inspector Executor Sparse Blas routines are used in the example:
    *
    *   Initialization/Destruction stage:
    *          mkl_sparse_d_create_csr
    *          mkl_sparse_destroy
    *
    *   Inspector stage:
    *          mkl_sparse_set_mv_hint  mkl_sparse_set_sv_hint
    *          mkl_sparse_set_mm_hint  mkl_sparse_set_sm_hint
    *          mkl_sparse_optimize
    *
    *   Executor stage:
    *          mkl_sparse_d_mv         mkl_sparse_d_trsv
    *          mkl_sparse_d_mm         mkl_sparse_d_trsm
    *
    * Consider the matrix A (see 'Sparse Storage Formats for Sparse BLAS Level 2
    * and Level 3 in the Intel oneMKL Reference Manual')
    *
    *                 |   1       -1      0   -3     0   |
    *                 |  -2        5      0    0     0   |
    *   A    =        |   0        0      4    6     4   |,
    *                 |  -4        0      2    7     0   |
    *                 |   0        8      0    0    -5   |
    *
    *  The matrix A is represented in a zero-based compressed sparse row (CSR) storage
    *  scheme with three arrays (see 'Sparse Matrix Storage Schemes' in the
    *   Intel oneMKL Reference Manual) as follows:
    *
    *         rowPtr  = ( 0        3     5        8       11    13 )
    *         columns = ( 0  1  3  0  1  2  3  4  0  2  3  1  4 )
    *         values  = ( 1 -1 -3 -2  5  4  6  4 -4  2  7  8 -5 )
    *
    ********************************************************************************/
    [Fact]
    public void SparseCSRTest()
    {
        //*******************************************************************************
        //     Declaration and initialization of parameters for sparse representation of
        //     the matrix A in the CSR format:
        //*******************************************************************************

        int m = 5;
        //int nnz = 13;
        int nrhs = 2;

        //*******************************************************************************
        //    Sparse representation of the matrix A
        //*******************************************************************************

        int[] rowPtr = { 0, 3, 5, 8, 11, 13 };

        int[] columns = { 0,      1,        3,
                              0,      1,
                                           2,   3,   4,
                              0,           2,   3,
                                      1,             4 };

        double[] values = { 1.0, -1.0,     -3.0,
                               -2.0,  5.0,
                                           4.0, 6.0, 4.0,
                               -4.0,       2.0, 7.0,
                                      8.0,          -5.0 };


        // Descriptor of main sparse matrix properties
        SparseBlas.matrix_descr descrA;

        // Structure with sparse matrix stored in CSR format
        //IntPtr csrA;

        //*******************************************************************************
        //    Declaration of local variables:
        //*******************************************************************************

        double[] x_m = { 1.0, 5.0, 3.0, 4.0, 2.0, 2.0, 10.0, 6.0, 8.0, 4.0 };
        double[] y_m = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        double[] x_v = { 3.0, 2.0, 5.0, 4.0, 1.0 };
        double[] y_v = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double[] tmp_v = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double alpha = 1.0, beta = 0.0;
        //int i;

        SparseStatus status;
        int exit_status = 0;

        //Console.Write( "\n EXAMPLE PROGRAM FOR CSR format routines from IE Sparse BLAS\n" );
        //Console.Write( "-------------------------------------------------------\n" );

        //*******************************************************************************
        //   Create CSR sparse matrix handle and analyze step
        //*******************************************************************************

        // Create handle with matrix stored in CSR format
        status = SparseBlas.create_csr(out IntPtr csrA, SparseIndexBase.SPARSE_INDEX_BASE_ZERO, m, m, rowPtr, rowPtr.Skip(1).Take(m).ToArray(), columns, values);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS) 
        //{
        //    Console.Write(" Error in mkl_sparse_d_create_csr: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //*******************************************************************************
        // First we set hints for the different operations before calling the
        // mkl_sparse_optimize() api which actually does the analyze step.  Not all
        // configurations have optimized steps, so the hint apis may return status
        // MKL_SPARSE_STATUS_NOT_SUPPORTED (=6) if no analysis stage is actually available
        // for that configuration.
        //*******************************************************************************

        //*******************************************************************************
        // Set hints for Task 1: Lower triangular transpose MV and SV solve with
        // non-unit diagonal
        //*******************************************************************************

        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_TRIANGULAR;
        descrA.mode = SparseFillMode.SPARSE_FILL_MODE_LOWER;
        descrA.diag = SparseDiagType.SPARSE_DIAG_NON_UNIT;

        status = SparseBlas.set_mv_hint(csrA, SparseOperation.SPARSE_OPERATION_TRANSPOSE, descrA, 1);
        if (status != SparseStatus.SPARSE_STATUS_SUCCESS && status != SparseStatus.SPARSE_STATUS_NOT_SUPPORTED)
        {
            //Console.Write(" Error in set hints for Task 1: mkl_sparse_set_mv_hint: %d \n", status);
        }

        status = SparseBlas.set_sv_hint(csrA, SparseOperation.SPARSE_OPERATION_TRANSPOSE, descrA, 1);
        if (status != SparseStatus.SPARSE_STATUS_SUCCESS && status != SparseStatus.SPARSE_STATUS_NOT_SUPPORTED)
        {
            //Console.Write(" Error in set hints for Task 1: mkl_sparse_set_mv_hint: %d \n", status);
        }

        //*******************************************************************************
        // Set hints for Task 2: Upper triangular non-transpose MV and SV solve with
        // unit diagonal
        //*******************************************************************************

        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_TRIANGULAR;
        descrA.mode = SparseFillMode.SPARSE_FILL_MODE_UPPER;
        descrA.diag = SparseDiagType.SPARSE_DIAG_UNIT;

        status = SparseBlas.set_mv_hint(csrA, SparseOperation.SPARSE_OPERATION_TRANSPOSE, descrA, 1);
        if (status != SparseStatus.SPARSE_STATUS_SUCCESS && status != SparseStatus.SPARSE_STATUS_NOT_SUPPORTED)
        {
            //Console.Write(" Error in set hints for Task 2: mkl_sparse_set_mv_hint: %d \n", status);
        }

        //*******************************************************************************
        // Set hints for Task 3: General matrix (transpose sparse) * dense MM  with
        // non-unit diagonal and column-major format
        //*******************************************************************************

        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_GENERAL;

        status = SparseBlas.set_mm_hint(csrA, SparseOperation.SPARSE_OPERATION_TRANSPOSE, descrA, SparseLayout.SPARSE_LAYOUT_COLUMN_MAJOR, nrhs, 1);
        if (status != SparseStatus.SPARSE_STATUS_SUCCESS && status != SparseStatus.SPARSE_STATUS_NOT_SUPPORTED)
        {
            //Console.Write(" Error in set hints for Task 3: mkl_sparse_set_mm_hint: %d \n", status);
        }

        //*******************************************************************************
        // Set hints for Task 4: General matrix sparse * dense MM  with
        // non-unit diagonal and row-major format
        //*******************************************************************************

        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_GENERAL;

        status = SparseBlas.set_mm_hint(csrA, SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, descrA, SparseLayout.SPARSE_LAYOUT_ROW_MAJOR, nrhs, 1);
        if (status != SparseStatus.SPARSE_STATUS_SUCCESS && status != SparseStatus.SPARSE_STATUS_NOT_SUPPORTED)
        {
            //Console.Write(" Error in set hints for Task 4: mkl_sparse_set_mm_hint: %d \n", status);
        }

        //*******************************************************************************
        // Analyze sparse matrix; choose proper kernels and workload balancing strategy
        //*******************************************************************************

        status = SparseBlas.optimize(csrA);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in mkl_sparse_optimize: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //*******************************************************************************
        //  Task 1: Obtain matrix-matrix multiply (L+D)' *x_v --> y_v
        //          and solve triangular system   (L+D)' *tmp_v = y_v
        //          Array tmp_v must be equal to the array x_v
        //*******************************************************************************

        //Console.Write("-------------------------------------------------------\n");
        //Console.Write("                                  \n");
        //Console.Write("   Task 1:                        \n");
        //Console.Write("   INPUT DATA FOR mkl_sparse_d_mv \n");
        //Console.Write("   WITH TRIANGULAR SPARSE MATRIX  \n");
        //Console.Write("   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta);
        //Console.Write("   SPARSE_OPERATION_TRANSPOSE     \n");
        //Console.Write("   Input vector                   \n");
        Assert.Equal(3.0, x_v[0]);
        Assert.Equal(2.0, x_v[1]);
        Assert.Equal(5.0, x_v[2]);
        Assert.Equal(4.0, x_v[3]);
        Assert.Equal(1.0, x_v[4]);

        // Create matrix descriptor
        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_TRIANGULAR;
        descrA.mode = SparseFillMode.SPARSE_FILL_MODE_LOWER;
        descrA.diag = SparseDiagType.SPARSE_DIAG_NON_UNIT;

        status = SparseBlas.mv(SparseOperation.SPARSE_OPERATION_TRANSPOSE, alpha, csrA, descrA, x_v, beta, y_v);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 1 mkl_sparse_d_mv: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_mv \n");
        Assert.Equal(-17.0, y_v[0]);
        Assert.Equal(18.0, y_v[1]);
        Assert.Equal(28.0, y_v[2]);
        Assert.Equal(28.0, y_v[3]);
        Assert.Equal(-5.0, y_v[4]);

        //Console.Write("   Solve triangular system   \n");
        //Console.Write("   with obtained             \n");
        //Console.Write("   right hand side           \n");

        status = SparseBlas.trsv(SparseOperation.SPARSE_OPERATION_TRANSPOSE, alpha, csrA, descrA, y_v, tmp_v);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 1 mkl_sparse_d_trsv: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_trsv \n");
        Assert.Equal(3.0, tmp_v[0]);
        Assert.Equal(2.0, tmp_v[1]);
        Assert.Equal(5.0, tmp_v[2]);
        Assert.Equal(4.0, tmp_v[3]);
        Assert.Equal(1.0, tmp_v[4]);
        //Console.Write("-------------------------------------------------------\n");

        //*******************************************************************************
        //  Task 2: Obtain matrix-matrix multiply (U+I)*x_v --> y_v
        //          and solve triangular system   (U+I)*tmp_v = y_v
        //          Array tmp_v must be equal to the array x_v
        //*******************************************************************************

        //Console.Write("                                  \n");
        //Console.Write("   Task 2:                        \n");
        //Console.Write("   INPUT DATA FOR mkl_sparse_d_mv \n");
        //Console.Write("   WITH TRIANGULAR SPARSE MATRIX  \n");
        //Console.Write("   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta);
        //Console.Write("   SPARSE_OPERATION_NON_TRANSPOSE \n");
        //Console.Write("   Input vector                   \n");
        Assert.Equal(3.0, x_v[0]);
        Assert.Equal(2.0, x_v[1]);
        Assert.Equal(5.0, x_v[2]);
        Assert.Equal(4.0, x_v[3]);
        Assert.Equal(1.0, x_v[4]);

        // Create matrix descriptor
        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_TRIANGULAR;
        descrA.mode = SparseFillMode.SPARSE_FILL_MODE_UPPER;
        descrA.diag = SparseDiagType.SPARSE_DIAG_UNIT;

        status = SparseBlas.mv(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, x_v, beta, y_v);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 2 mkl_sparse_d_mv: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_mv \n");
        Assert.Equal(-11.0, y_v[0]);
        Assert.Equal(2.0, y_v[1]);
        Assert.Equal(33.0, y_v[2]);
        Assert.Equal(4.0, y_v[3]);
        Assert.Equal(1.0, y_v[4]);

        //Console.Write("   Solve triangular system   \n");
        //Console.Write("   with obtained             \n");
        //Console.Write("   right hand side           \n");

        status = SparseBlas.trsv(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, y_v, tmp_v);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 2 mkl_sparse_d_trsv: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_trsv \n");
        Assert.Equal(3.0, tmp_v[0]);
        Assert.Equal(2.0, tmp_v[1]);
        Assert.Equal(5.0, tmp_v[2]);
        Assert.Equal(4.0, tmp_v[3]);
        Assert.Equal(1.0, tmp_v[4]);
        //Console.Write("-------------------------------------------------------\n");

        //*******************************************************************************
        //  Task 3: Obtain matrix-matrix multiply A' *x_m --> y_m
        //          A - zero-based indexing,
        //          x_m - column major ordering
        //*******************************************************************************

        //Console.Write("                                  \n");
        //Console.Write("   Task 3:                        \n");
        //Console.Write("   INPUT DATA FOR mkl_sparse_d_mm \n");
        //Console.Write("   WITH GENERAL SPARSE MATRIX     \n");
        //Console.Write("   COLUMN MAJOR ORDERING for RHS  \n");
        //Console.Write("   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta);
        //Console.Write("   SPARSE_OPERATION_TRANSPOSE     \n");
        //Console.Write("   Input vectors                  \n");
        Assert.Equal(1.0, x_m[0]);
        Assert.Equal(5.0, x_m[1]);
        Assert.Equal(3.0, x_m[2]);
        Assert.Equal(4.0, x_m[3]);
        Assert.Equal(2.0, x_m[4]);
        Assert.Equal(2.0, x_m[5]);
        Assert.Equal(10.0, x_m[6]);
        Assert.Equal(6.0, x_m[7]);
        Assert.Equal(8.0, x_m[8]);
        Assert.Equal(4.0, x_m[9]);

        // Create matrix descriptor
        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_GENERAL;
        descrA.diag = SparseDiagType.SPARSE_DIAG_NON_UNIT;

        // note that column-major format implies  ldx = m, ldy = m
        status = SparseBlas.mm(SparseOperation.SPARSE_OPERATION_TRANSPOSE, alpha, csrA, descrA, SparseLayout.SPARSE_LAYOUT_COLUMN_MAJOR, x_m, nrhs, m, beta, y_m, m);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 3 mkl_sparse_d_mm: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_mm \n");
        Assert.Equal(-25.0, y_m[0]);
        Assert.Equal(40.0, y_m[1]);
        Assert.Equal(20.0, y_m[2]);
        Assert.Equal(43.0, y_m[3]);
        Assert.Equal(2.0, y_m[4]);
        Assert.Equal(-50.0, y_m[5]);
        Assert.Equal(80.0, y_m[6]);
        Assert.Equal(40.0, y_m[7]);
        Assert.Equal(86.0, y_m[8]);
        Assert.Equal(4.0, y_m[9]);
        //Console.Write("-------------------------------------------------------\n");

        //*******************************************************************************
        //  Task 4: Obtain matrix-matrix multiply A*x_m --> y_m
        //          A - zero-based indexing,
        //          x_m - row major ordering
        //*******************************************************************************

        //Console.Write("                                  \n");
        //Console.Write("   Task 4:                        \n");
        //Console.Write("   INPUT DATA FOR mkl_sparse_d_mm \n");
        //Console.Write("   WITH GENERAL SPARSE MATRIX     \n");
        //Console.Write("   ROW MAJOR ORDERING for RHS     \n");
        //Console.Write("   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta);
        //Console.Write("   SPARSE_OPERATION_TRANSPOSE     \n");
        //Console.Write("   Input vectors                  \n");
        Assert.Equal(1.0, x_m[0]);
        Assert.Equal(5.0, x_m[1]);
        Assert.Equal(3.0, x_m[2]);
        Assert.Equal(4.0, x_m[3]);
        Assert.Equal(2.0, x_m[4]);
        Assert.Equal(2.0, x_m[5]);
        Assert.Equal(10.0, x_m[6]);
        Assert.Equal(6.0, x_m[7]);
        Assert.Equal(8.0, x_m[8]);
        Assert.Equal(4.0, x_m[9]);

        // Create matrix descriptor
        descrA.type = SparseMatrixType.SPARSE_MATRIX_TYPE_GENERAL;
        descrA.diag = SparseDiagType.SPARSE_DIAG_NON_UNIT;

        // note that row-major format implies  ldx = nrhs, ldy = nrhs
        status = SparseBlas.mm(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, SparseLayout.SPARSE_LAYOUT_ROW_MAJOR, x_m, nrhs, nrhs, beta, y_m, nrhs);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in Task 4 mkl_sparse_d_mm: %d \n", status);
        //    exit_status = 1;
        //    goto exit;
        //}

        //Console.Write("   OUTPUT DATA FOR mkl_sparse_d_mm \n");
        Assert.Equal(-32.0, y_m[0]);
        Assert.Equal(-17.0, y_m[1]);
        Assert.Equal(13.0, y_m[2]);
        Assert.Equal(10.0, y_m[3]);
        Assert.Equal(100.0, y_m[4]);
        Assert.Equal(60.0, y_m[5]);
        Assert.Equal(70.0, y_m[6]);
        Assert.Equal(26.0, y_m[7]);
        Assert.Equal(-16.0, y_m[8]);
        Assert.Equal(12.0, y_m[9]);
        //Console.Write("-------------------------------------------------------\n");

        //exit:

        // Release matrix handle and deallocate matrix
        status = SparseBlas.destroy(csrA);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);
        //if (status != SparseBlas.STATUS.SPARSE_STATUS_SUCCESS)
        //{
        //    Console.Write(" Error in matrix deallocation: %d \n", status);
        //    exit_status = 1;
        //}

        Assert.True(exit_status == 0);
    }

    /*******************************************************************************
    *   Content : Intel(R) oneAPI Math Kernel Library (oneMKL) IE Sparse BLAS C
    *             example for mkl_sparse_spmm
    *
    ********************************************************************************
    *
    * Consider the matrix A
    *
    *                 |  10     11      0     0     0   |
    *                 |   0      0     12    13     0   |
    *   A    =        |  15      0      0     0    14   |,
    *                 |   0     16     17     0     0   |
    *                 |   0      0      0    18    19   |
    *
    * and diagonal matrix B
    *
    *                 |   5      0      0     0     0   |
    *                 |   0      6      0     0     0   |
    *   B    =        |   0      0      7     0     0   |.
    *                 |   0      0      0     8     0   |
    *                 |   0      0      0     0     9   |
    *
    *  Both matrices A and B are stored in a zero-based compressed sparse row (CSR) storage
    *  scheme with three arrays (see 'Sparse Matrix Storage Schemes' in the
    *   Intel oneMKL Developer Reference) as follows:
    *
    *           values_A = ( 10  11  12  13  15  14  16  17  18  19 )
    *          columns_A = (  0   1   2   3   0   4   1   2   3   4 )
    *         rowIndex_A = (  0       2       4       6       8      10 )
    *
    *           values_B = ( 5  6  7  8  9  )
    *          columns_B = ( 0  1  2  3  4  )
    *         rowIndex_B = ( 0  1  2  3  4  5 )
    *
    *  The example computes two scalar products :
    *
    *         < (A*B)*x ,       y > = left,   using MKL_SPARSE_SPMM and CBLAS_DDOT.
    *         <     B*x , (A^t)*y > = right,  using MKL_SPARSE_D_MV and CBLAS_DDOT.
    *
    *         These products should result in the same value. To obtain matrix C,
    *         use MKL_SPARSE_D_EXPORT_CSR and print the result.
    *
    ********************************************************************************/
    [Fact]
    public void SparseSpmmExportCSRTest()
    {
        int M = 5;
        int NNZ = 10;
       
        //IntPtr csrA;
        //IntPtr csrB;
        //IntPtr csrC;

        /* Declaration of values */
        /* Allocation of memory */
        double[] values_A = new double[NNZ];
        int[] columns_A = new int[NNZ];
        int[] rowIndex_A = new int[M + 1];

        double[] values_B = new double[M];
        int[] columns_B = new int[M];
        int[] rowIndex_B = new int[M + 1];

        double[] x = new double[M];
        double[] y = new double[M];
        double[] rslt_mv = new double[M];
        double[] rslt_mv_trans = new double[M];

        double left, right, residual;

        int i;
        SparseStatus status;
        SparseBlas.matrix_descr descr_type_gen;

        /* Set values of the variables*/
        descr_type_gen.type = SparseMatrixType.SPARSE_MATRIX_TYPE_GENERAL;
        /* The following descriptor fields are not applicable for matrix type
        * SPARSE_MATRIX_TYPE_GENERAL but must be specified for other matrix types: */
        descr_type_gen.mode = SparseFillMode.SPARSE_FILL_MODE_FULL;
        descr_type_gen.diag = SparseDiagType.SPARSE_DIAG_NON_UNIT;
        
        // Matrix A 
        for (i = 0; i < NNZ; i++)
        {
            values_A[i] = i + 10;
        }
        for (i = 0; i < NNZ; i++)
        {
            columns_A[i] = i % 5;
        }
        rowIndex_A[0] = 0;
        for (i = 1; i < M + 1; i++)
        {
            rowIndex_A[i] = rowIndex_A[i - 1] + 2;
        }

        // Matrix B
        for (i = 0; i < M; i++)
        {
            values_B[i] = i + 5;
        }
        for (i = 0; i < M; i++)
        {
            columns_B[i] = i % 5;
        }
        for (i = 0; i < M + 1; i++)
        {
            rowIndex_B[i] = i;
        }

        // Vectors x and y
        for (i = 0; i<M; i++)
        {
            x[i] = 1.0; y[i] = 1.0;
        }

        /* Printing usable data */
        //Console.Write("\n\n_______________Example program for MKL_SPARSE_SPMM_________________\n\n");
        //Console.Write(" COMPUTE  A * B = C, where matrices are stored in CSR format\n");
        //Console.Write("\n MATRIX A:\nrow# : (value, column) (value, column)\n");
        Assert.Equal(10.0, values_A[0]);
        Assert.Equal(0, columns_A[0]);
        Assert.Equal(11.0, values_A[1]);
        Assert.Equal(1, columns_A[1]);
        Assert.Equal(12.0, values_A[2]);
        Assert.Equal(2, columns_A[2]);
        Assert.Equal(13.0, values_A[3]);
        Assert.Equal(3, columns_A[3]);
        Assert.Equal(14.0, values_A[4]);
        Assert.Equal(4, columns_A[4]);
        Assert.Equal(15.0, values_A[5]);
        Assert.Equal(0, columns_A[5]);
        Assert.Equal(16.0, values_A[6]);
        Assert.Equal(1, columns_A[6]);
        Assert.Equal(17.0, values_A[7]);
        Assert.Equal(2, columns_A[7]);
        Assert.Equal(18.0, values_A[8]);
        Assert.Equal(3, columns_A[8]);
        Assert.Equal(19.0, values_A[9]);
        Assert.Equal(4, columns_A[9]);
        //Console.Write("\n MATRIX B:\nrow# : (value, column)\n");
        Assert.Equal(5.0, values_B[0]);
        Assert.Equal(0, columns_B[0]);
        Assert.Equal(6.0, values_B[1]);
        Assert.Equal(1, columns_B[1]);
        Assert.Equal(7.0, values_B[2]);
        Assert.Equal(2, columns_B[2]);
        Assert.Equal(8.0, values_B[3]);
        Assert.Equal(3, columns_B[3]);
        Assert.Equal(9.0, values_B[4]);
        Assert.Equal(4, columns_B[4]);
        //Console.Write("\n Check the resultant matrix C, using two scalar products\n");
        //Console.Write(" (values of these scalar products must match).\n");

        /* Prepare arrays, which are related to matrices.
           Create handles for matrices A and B stored in CSR format */

        status = SparseBlas.create_csr(out IntPtr csrA, SparseIndexBase.SPARSE_INDEX_BASE_ZERO, M, M, rowIndex_A, rowIndex_A.Skip(1).Take(M).ToArray(), columns_A, values_A);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.create_csr(out IntPtr csrB, SparseIndexBase.SPARSE_INDEX_BASE_ZERO, M, M, rowIndex_B, rowIndex_B.Skip(1).Take(M).ToArray(), columns_B, values_B);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        /* Compute C = A * B  */

        status = SparseBlas.spmm(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, out IntPtr csrC);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        /* Analytic Routines for MKL_SPARSE_D_MV.
           HINTS: provides estimate of number and type of upcoming matrix-vector operations
           OPTIMIZE: analyze sparse matrix; choose proper kernels and workload balancing strategy */

        status = SparseBlas.set_mv_hint(csrA, SparseOperation.SPARSE_OPERATION_TRANSPOSE, descr_type_gen, 1);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.set_mv_hint(csrB, SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen, 1);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.set_mv_hint(csrC, SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen, 1);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.optimize(csrA);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.optimize(csrB);
       
        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.optimize(csrC);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        /* Execution Routines */

        /* Step 1:
                  Need to compute the following variables:
                         rslt_mv = C * x
                            left = <rslt_mv, y>              */

        status = SparseBlas.mv(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrC, descr_type_gen, x, 0.0, rslt_mv);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        left = Blas.dot(M, rslt_mv, 1, y, 1);

        /* Step 2:
                  Need to compute the following variables:
                   rslt_mv       =     B * x
                   rslt_mv_trans = (A)^t * y
                           right = <rslt_mv, rslt_mv_trans>  */

        status = SparseBlas.mv(SparseOperation.SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrB, descr_type_gen, x, 0.0, rslt_mv);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        status = SparseBlas.mv(SparseOperation.SPARSE_OPERATION_TRANSPOSE, 1.0, csrA, descr_type_gen, y, 0.0, rslt_mv_trans);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        right = Blas.dot(M, rslt_mv, 1, rslt_mv_trans, 1);

        /* Step 3:
                  Compare values obtained for left and right  */

        residual = Math.Abs(left - right) / (Math.Abs(left) + 1);

        //Console.Write("\n The difference between < C*x , y > and < B*x , (A^t)*y > = %g,\n", residual);
        //Console.Write(" which means that MKL_SPARSE_SPMM arrived correct at a solution.\n");

        Assert.Equal(0.0, residual);

        /* Printing OUTPUT DATA */
        status = SparseBlas.export_csr(csrC, out _, out _, out _, out _, out _, out int[] columns_C, out double[] values_C);
        
        //status = SparseBlas.D_export_csr(csrC, ref indexing, ref rows, ref cols, pointerB_C, pointerE_C, columns_C, values_C);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        //Console.Write("\n RESULTANT MATRIX C:\nrow# : (value, column) (value, column)\n");
        Assert.Equal(50.0, values_C[0]);
        Assert.Equal(0, columns_C[0]);
        Assert.Equal(66.0, values_C[1]);
        Assert.Equal(1, columns_C[1]);
        Assert.Equal(84.0, values_C[2]);
        Assert.Equal(2, columns_C[2]);
        Assert.Equal(104.0, values_C[3]);
        Assert.Equal(3, columns_C[3]);
        Assert.Equal(126.0, values_C[4]);
        Assert.Equal(4, columns_C[4]);
        Assert.Equal(75.0, values_C[5]);
        Assert.Equal(0, columns_C[5]);
        Assert.Equal(96.0, values_C[6]);
        Assert.Equal(1, columns_C[6]);
        Assert.Equal(119.0, values_C[7]);
        Assert.Equal(2, columns_C[7]);
        Assert.Equal(144.0, values_C[8]);
        Assert.Equal(3, columns_C[8]);
        Assert.Equal(171.0, values_C[9]);
        Assert.Equal(4, columns_C[9]);
        //Console.Write("_____________________________________________________________________  \n");

        /* Deallocate memory */
        //memory_free:

        //Release matrix handle. Not necessary to deallocate arrays for which we don't allocate memory: values_C, columns_C, pointerB_C, and pointerE_C.
        //These arrays will be deallocated together with csrC structure.

        status = SparseBlas.destroy(csrC);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        //if (mkl_sparse_destroy(csrC) != SPARSE_STATUS_SUCCESS)
        //{
        //    printf(" Error after MKL_SPARSE_DESTROY, csrC \n"); fflush(0); status = 1;
        //}

        //Deallocate arrays for which we allocate memory ourselves.
        //mkl_free(rslt_mv_trans); mkl_free(rslt_mv); mkl_free(x); mkl_free(y);

        //Release matrix handle and deallocate arrays for which we allocate memory ourselves.

        status = SparseBlas.destroy(csrA);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        //if (mkl_sparse_destroy(csrA) != SPARSE_STATUS_SUCCESS)
        //{
        //    printf(" Error after MKL_SPARSE_DESTROY, csrA \n"); fflush(0); status = 1;
        //}
        //mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);

        status = SparseBlas.destroy(csrB);

        Assert.Equal(SparseStatus.SPARSE_STATUS_SUCCESS, status);

        //if (mkl_sparse_destroy(csrB) != SPARSE_STATUS_SUCCESS)
        //{
        //    printf(" Error after MKL_SPARSE_DESTROY, csrB \n"); fflush(0); status = 1;
        //}
        //mkl_free(values_B); mkl_free(columns_B); mkl_free(rowIndex_B);
    }
}

