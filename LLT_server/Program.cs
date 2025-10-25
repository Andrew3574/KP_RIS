// Program.cs
using MPI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DistributedLLTSolver
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var environment = new MPI.Environment(ref args))
            {
                Intracommunicator comm = MPI.Communicator.world;
                
                if (comm.Rank == 0)
                {
                    Console.WriteLine($"Starting distributed SLAU solver on {comm.Size} processes");
                    Console.WriteLine($"Maximum number of unknowns: 50000");
                }

                if (args.Length >= 1 && args[0] == "test")
                {
                    int size = args.Length >= 2 ? int.Parse(args[1]) : 100;
                    RunTest(comm, size);
                }
                else if (args.Length >= 3)
                {
                    string matrixFile = args[0];
                    string vectorFile = args[1];
                    string nodesFile = args[2];
                    
                    SolveFromFiles(comm, matrixFile, vectorFile, nodesFile);
                }
                else
                {
                    if (comm.Rank == 0)
                    {
                        Console.WriteLine("Usage: mpiexec -n <processes> dotnet run -- <matrix> <vector> <nodes>");
                        Console.WriteLine("Or for test: mpiexec -n 4 dotnet run -- test 1000");
                    }
                }
            }
        }

        static void RunTest(Intracommunicator comm, int size)
        {
            Matrix<double>? A = null;
            Vector<double>? b = null;
            
            if (comm.Rank == 0)
            {
                A = GenerateSPDMatrix(size);
                b = Vector<double>.Build.Random(size);
                
                Console.WriteLine($"Test system: {size} x {size}");
                Console.WriteLine("Matrix generated (symmetric positive definite)");
            }

            // Distributed solution using LLT method
            var watch = System.Diagnostics.Stopwatch.StartNew();
            Vector<double> xLLT = DistributedLLTSolver.Solve(comm, A!, b!);
            watch.Stop();
            
            // Gather full solution on process 0
            Vector<double>? fullX = GatherSolution(comm, xLLT, size);
            
            if (comm.Rank == 0 && A != null && b != null && fullX != null)
            {
                Console.WriteLine($"LLT method: {watch.ElapsedMilliseconds} ms");
                
                // Solution verification
                Vector<double> residual = A * fullX - b;
                double error = residual.L2Norm();
                Console.WriteLine($"LLT residual: {error}");
                
                // Comparison with Gaussian method (only for small systems)
                if (size <= 1000)
                {
                    watch.Restart();
                    Vector<double> xGauss = GaussianSolver.Solve(A, b);
                    watch.Stop();
                    Console.WriteLine($"Gaussian method: {watch.ElapsedMilliseconds} ms");
                    
                    Vector<double> residualGauss = A * xGauss - b;
                    double errorGauss = residualGauss.L2Norm();
                    Console.WriteLine($"Gaussian residual: {errorGauss}");
                    
                    // Solutions comparison
                    double diff = (fullX - xGauss).L2Norm();
                    Console.WriteLine($"Difference between solutions: {diff}");
                }
                else
                {
                    Console.WriteLine("Gaussian method skipped for large systems (too slow)");
                }
            }
        }

        // Method for gathering solution from all processes
        private static Vector<double>? GatherSolution(Intracommunicator comm, Vector<double> localX, int totalSize)
        {
            if (comm.Rank == 0)
            {
                var fullX = Vector<double>.Build.Dense(totalSize);
                int worldSize = comm.Size;
                int rowsPerProcess = totalSize / worldSize;
                int remainder = totalSize % worldSize;
                
                // Copy local solution of process 0
                int start0 = 0;
                int end0 = rowsPerProcess + (0 < remainder ? 1 : 0);
                for (int i = 0; i < end0 - start0; i++)
                {
                    fullX[i] = localX[i];
                }
                
                // Receive solutions from other processes element by element
                for (int source = 1; source < worldSize; source++)
                {
                    int start = source * rowsPerProcess + Math.Min(source, remainder);
                    int end = start + rowsPerProcess + (source < remainder ? 1 : 0);
                    int count = end - start;
                    
                    for (int i = 0; i < count; i++)
                    {
                        double value = comm.Receive<double>(source, i);
                        fullX[start + i] = value;
                    }
                }
                
                return fullX;
            }
            else
            {
                // Send local solution to process 0 element by element
                for (int i = 0; i < localX.Count; i++)
                {
                    comm.Send(localX[i], 0, i);
                }
                return null; // On other processes return null
            }
        }

        static void SolveFromFiles(Intracommunicator comm, string matrixFile, string vectorFile, string nodesFile)
        {
            if (comm.Rank == 0)
            {
                Console.WriteLine("Reading data from files...");
                Console.WriteLine($"Matrix: {matrixFile}");
                Console.WriteLine($"Vector: {vectorFile}");
                Console.WriteLine($"Nodes: {nodesFile}");
            }
        }

        static Matrix<double> GenerateSPDMatrix(int size)
        {
            var random = new Random();
            var A = Matrix<double>.Build.Dense(size, size);
            
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double value = random.NextDouble() * 2.0 - 1.0;
                    A[i, j] = value;
                    A[j, i] = value;
                }
                A[i, i] += size; // ensuring diagonal dominance
            }
            
            return A;
        }
    }
}