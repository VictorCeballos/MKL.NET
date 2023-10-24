namespace MKL.NET.Test; 

using System;
using Xunit;
using MKLNET;


public class SparseSolverTest
{
    [Fact]
    public void SymmetricLinearSystemTest()
    {
        /* Matrix data. */
        int n = 8;
        int[] ia = new int[] { 0, 4, 7, 9, 11, 14, 16, 17, 18 };
        int[] ja = new int[] { 0,    2,       5, 6,
                                    1, 2,    4,
                                        2,             7,
                                        3,       6,
                                            4, 5, 6,
                                                5,    7,
                                                    6,
                                                    7 };
        double[] a = new double[] { 7.0,      1.0,           2.0, 7.0,
                                        -4.0, 8.0,      2.0,
                                                1.0,                     5.0,
                                                    7.0,           9.0,
                                                        5.0, 1.0, 5.0,
                                                                0.0,      5.0,
                                                                    11.0,
                                                                        5.0 };

        int nnz = ia[n];
        int mtype = -2;        /* Real symmetric matrix */

        /* RHS and solution vectors. */
        double[] b = new double[8];
        double[] x = new double[8];
        int nrhs = 1;          /* Number of right hand sides. */

        /* Internal solver memory pointer pt,                  */
        /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
        /* or void *pt[64] should be OK on both architectures  */
        IntPtr[] pt = new IntPtr[64];

        /* Pardiso control parameters. */
        int[] iparm = new int[64];
        //Span<double> dparm = new double[64];
        int maxfct, mnum, phase, msglvl;
        //int error;
        //int solver;

        /* Number of processors. */
        int num_procs;

        /* Auxiliary variables. */
        int i;

        double[] ddum = new double[1];  /* Double dummy */
        int[] idum = new int[1];        /* Integer dummy. */

        /* -------------------------------------------------------------------- */
        /* ..  Setup Pardiso control parameters.                                */
        /* -------------------------------------------------------------------- */

        //error = 0;
        //solver = 0; /* use sparse direct solver */
        SparseSolver.pardisoinit(pt, ref mtype, iparm);

        /*try
        {
            Assert.Equal(0, error);
        }
        catch (Xunit.Sdk.EqualException ex)
        {
            if (error == -10)
            {
                throw new Xunit.Sdk.XunitException($"No license file found: {error}\n{ex}");
            }
            if (error == -11)
            {
                throw new Xunit.Sdk.XunitException($"License is expired: {error}\n{ex}");
            }
            if (error == -12)
            {
                throw new Xunit.Sdk.XunitException($"Wrong username or hostname: {error}\n{ex}");
            }
            throw new Xunit.Sdk.XunitException($"\nERROR license check: {error}\n{ex}");
        }*/

        //Debug.WriteLine("[PARDISO]: License check was successful ... \n");

        /* Numbers of processors, value of OMP_NUM_THREADS */
        num_procs = 1;
        iparm[2] = num_procs;

        maxfct = 1;     /* Maximum number of numerical factorizations.  */
        mnum = 1;       /* Which factorization to use. */

        msglvl = 1;     /* Print statistical information  */
        //error = 0;      /* Initialize error flag */

        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
        /*     notation.                                                        */
        /* -------------------------------------------------------------------- */

        for (i = 0; i < n + 1; i++)
        {
            ia[i] += 1;
        }
        for (i = 0; i < nnz; i++)
        {
            ja[i] += 1;
        }

        /* Set right hand side to one. */
        for (i = 0; i < n; i++)
        {
            b[i] = i;
        }

        /* -------------------------------------------------------------------- */
        /* ..  Reordering and Symbolic Factorization.  This step also allocates */
        /*     all memory that is necessary for the factorization.              */
        /* -------------------------------------------------------------------- */

        phase = 11;

        SparseSolver.pardiso(pt, ref maxfct, ref mnum, ref mtype, ref phase, ref n, a, ia, ja, idum, ref nrhs, iparm, ref msglvl, ddum, ddum, out int error);

        try
        {
            Assert.Equal(0, error);
        }
        catch (Xunit.Sdk.EqualException ex)
        {
            throw new Xunit.Sdk.XunitException($"\nERROR during symbolic factorization: {error}\n{ex}");
        }

        //Debug.WriteLine("\nReordering completed ... ");
        Assert.Equal(32, iparm[17]); // number of nonzeros in factors
        Assert.Equal(0, iparm[18]); // number of factorization MFLOPS

        /* -------------------------------------------------------------------- */
        /* ..  Numerical factorization.                                         */
        /* -------------------------------------------------------------------- */

        phase = 22;
        iparm[32] = 1; /* compute determinant */

        SparseSolver.pardiso(pt, ref maxfct, ref mnum, ref mtype, ref phase, ref n, a, ia, ja, idum, ref nrhs, iparm, ref msglvl, ddum, ddum, out error);

        try
        {
            Assert.Equal(0, error);
        }
        catch (Xunit.Sdk.EqualException ex)
        {
            throw new Xunit.Sdk.XunitException($"\nERROR during numerical factorization: {error}\n{ex}");
        }

        //Debug.WriteLine("\nFactorization completed ...\n ");

        /* -------------------------------------------------------------------- */
        /* ..  Back substitution and iterative refinement.                      */
        /* -------------------------------------------------------------------- */

        phase = 33;
        iparm[7] = 1;       /* Max numbers of iterative refinement steps. */

        SparseSolver.pardiso(pt, ref maxfct, ref mnum, ref mtype, ref phase, ref n, a, ia, ja, idum, ref nrhs, iparm, ref msglvl, b, x, out error);

        try
        {
            Assert.Equal(0, error);
        }
        catch (Xunit.Sdk.EqualException ex)
        {
            throw new Xunit.Sdk.XunitException($"\nERROR during solution: {error}\n{ex}");
        }

        //Debug.WriteLine("\nSolve completed ... ");
        //Debug.WriteLine("\nThe solution of the system is: ");
        Assert.Equal(-0.21155972839670001, x[0]);
        Assert.Equal(-0.28509591636623671, x[1]);
        Assert.Equal(-0.20019684603985574, x[2]);
        Assert.Equal(0.362865768694844440, x[3]);
        Assert.Equal(0.730595551426949430, x[4]);
        Assert.Equal(0.661692064966565630, x[5]);
        Assert.Equal(0.051104402126232122, x[6]);
        Assert.Equal(0.938504781073289940, x[7]);

        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix back to 0-based C-notation.                       */
        /* -------------------------------------------------------------------- */

        for (i = 0; i < n + 1; i++)
        {
            ia[i] -= 1;
        }
        for (i = 0; i < nnz; i++)
        {
            ja[i] -= 1;
        }

        /* -------------------------------------------------------------------- */
        /* ..  Termination and release of memory.                               */
        /* -------------------------------------------------------------------- */

        phase = -1;                 /* Release internal memory. */

        SparseSolver.pardiso(pt, ref maxfct, ref mnum, ref mtype, ref phase, ref n, ddum, ia, ja, idum, ref nrhs, iparm, ref msglvl, ddum, ddum, out error);

        try
        {
            Assert.Equal(0, error);
        }
        catch (Xunit.Sdk.EqualException ex)
        {
            throw new Xunit.Sdk.XunitException($"\nERROR during termination: {error}\n{ex}");
        }
    }
}

