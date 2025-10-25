// DistributedLLTSolver.cs
using MPI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DistributedLLTSolver
{
    public static class DistributedLLTSolver
    {
        public static Vector<double> Solve(Intracommunicator comm, Matrix<double> A, Vector<double> b)
        {
            int worldSize = comm.Size;
            int rank = comm.Rank;
            
            int n;
            if (rank == 0)
            {
                n = A.RowCount;
            }
            else
            {
                n = 0;
            }
            
            comm.Broadcast(ref n, 0);
            
            var localData = DistributeMatrixSimple(comm, A, n);
            var localB = DistributeVectorSimple(comm, b, n);
            
            if (rank == 0)
            {
                Console.WriteLine("Starting LLT decomposition...");
            }
            
            LLTDecompositionSimple(comm, localData, n);
            
            if (rank == 0)
            {
                Console.WriteLine("LLT decomposition completed");
            }
            
            var y = ForwardSubstitution(comm, localData, localB, n);
            var x = BackwardSubstitution(comm, localData, y, n);
            
            // Return local part of solution
            return x;
        }
        
        private static Matrix<double> DistributeMatrixSimple(Intracommunicator comm, Matrix<double> A, int n)
        {
            int rank = comm.Rank;
            int worldSize = comm.Size;
            
            int rowsPerProcess = n / worldSize;
            int remainder = n % worldSize;
            
            int startRow = rank * rowsPerProcess + Math.Min(rank, remainder);
            int endRow = startRow + rowsPerProcess + (rank < remainder ? 1 : 0);
            int localRows = endRow - startRow;
            
            var localA = Matrix<double>.Build.Dense(localRows, n);
            
            if (rank == 0)
            {
                for (int i = 0; i < localRows; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        localA[i, j] = A[i, j];
                    }
                }
                
                for (int dest = 1; dest < worldSize; dest++)
                {
                    int destStart = dest * rowsPerProcess + Math.Min(dest, remainder);
                    int destRows = rowsPerProcess + (dest < remainder ? 1 : 0);
                    
                    for (int i = 0; i < destRows; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            double value = A[destStart + i, j];
                            comm.Send(value, dest, i * n + j);
                        }
                    }
                }
            }
            else
            {
                int localRowsForThisProcess = localRows;
                
                for (int i = 0; i < localRowsForThisProcess; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double value = comm.Receive<double>(0, i * n + j);
                        localA[i, j] = value;
                    }
                }
            }
            
            return localA;
        }
        
        private static Vector<double> DistributeVectorSimple(Intracommunicator comm, Vector<double> b, int n)
        {
            int rank = comm.Rank;
            int worldSize = comm.Size;
            
            int rowsPerProcess = n / worldSize;
            int remainder = n % worldSize;
            
            int startRow = rank * rowsPerProcess + Math.Min(rank, remainder);
            int endRow = startRow + rowsPerProcess + (rank < remainder ? 1 : 0);
            int localRows = endRow - startRow;
            
            var localB = Vector<double>.Build.Dense(localRows);
            
            if (rank == 0)
            {
                for (int i = 0; i < localRows; i++)
                {
                    localB[i] = b[i];
                }
                
                for (int dest = 1; dest < worldSize; dest++)
                {
                    int destStart = dest * rowsPerProcess + Math.Min(dest, remainder);
                    int destRows = rowsPerProcess + (dest < remainder ? 1 : 0);
                    
                    for (int i = 0; i < destRows; i++)
                    {
                        double value = b[destStart + i];
                        comm.Send(value, dest, i);
                    }
                }
            }
            else
            {
                int localRowsForThisProcess = localRows;
                
                for (int i = 0; i < localRowsForThisProcess; i++)
                {
                    double value = comm.Receive<double>(0, i);
                    localB[i] = value;
                }
            }
            
            return localB;
        }
        
        private static void LLTDecompositionSimple(Intracommunicator comm, Matrix<double> localL, int n)
        {
            int rank = comm.Rank;
            int worldSize = comm.Size;
            
            int rowsPerProcess = n / worldSize;
            int remainder = n % worldSize;
            
            // Temporary array for storing k-th row of matrix L
            double[] currentRow = new double[n];
            
            for (int k = 0; k < n; k++)
            {
                int kOwner = FindProcessForRow(k, rowsPerProcess, remainder, worldSize);
                
                // Calculate k-th row of matrix L
                if (rank == kOwner)
                {
                    int localK = GlobalToLocalIndex(k, rowsPerProcess, remainder, kOwner);
                    
                    // Calculate L[k,k]
                    double sum = 0.0;
                    for (int j = 0; j < k; j++)
                    {
                        sum += localL[localK, j] * localL[localK, j];
                    }
                    localL[localK, k] = Math.Sqrt(localL[localK, k] - sum);
                    currentRow[k] = localL[localK, k];
                    
                    // Calculate L[k,j] for j = k+1..n-1
                    for (int j = k + 1; j < n; j++)
                    {
                        sum = 0.0;
                        for (int m = 0; m < k; m++)
                        {
                            sum += localL[localK, m] * currentRow[m];
                        }
                        localL[localK, j] = (localL[localK, j] - sum) / localL[localK, k];
                        currentRow[j] = localL[localK, j];
                    }
                }
                
                // Broadcast k-th row to all processes
                comm.Broadcast(ref currentRow, kOwner);
                
                // Update rows i > k in current process
                for (int i = 0; i < localL.RowCount; i++)
                {
                    int globalI = LocalToGlobalIndex(i, rank, rowsPerProcess, remainder);
                    if (globalI > k)
                    {
                        // Calculate L[i,k]
                        double sum = 0.0;
                        for (int j = 0; j < k; j++)
                        {
                            sum += localL[i, j] * currentRow[j];
                        }
                        localL[i, k] = (localL[i, k] - sum) / currentRow[k];
                    }
                }
            }
        }
        
        private static Vector<double> ForwardSubstitution(Intracommunicator comm, Matrix<double> localL, 
            Vector<double> localB, int n)
        {
            int rank = comm.Rank;
            int worldSize = comm.Size;
            
            int rowsPerProcess = n / worldSize;
            int remainder = n % worldSize;
            
            var localY = Vector<double>.Build.Dense(localL.RowCount);
            var globalY = Vector<double>.Build.Dense(n);
            
            for (int i = 0; i < n; i++)
            {
                int owner = FindProcessForRow(i, rowsPerProcess, remainder, worldSize);
                double yi = 0.0;
                
                if (rank == owner)
                {
                    int localI = GlobalToLocalIndex(i, rowsPerProcess, remainder, owner);
                    if (localI >= 0 && localI < localL.RowCount)
                    {
                        double sum = 0.0;
                        for (int j = 0; j < i; j++)
                        {
                            sum += localL[localI, j] * globalY[j];
                        }
                        
                        yi = (localB[localI] - sum) / localL[localI, i];
                        globalY[i] = yi;
                    }
                }
                
                double yiToBroadcast = yi;
                comm.Broadcast(ref yiToBroadcast, owner);
                globalY[i] = yiToBroadcast;
                
                for (int loc = 0; loc < localL.RowCount; loc++)
                {
                    int globalIdx = LocalToGlobalIndex(loc, rank, rowsPerProcess, remainder);
                    if (globalIdx == i)
                    {
                        localY[loc] = yiToBroadcast;
                    }
                }
            }
            
            return localY;
        }
        
        private static Vector<double> BackwardSubstitution(Intracommunicator comm, Matrix<double> localL, 
            Vector<double> y, int n)
        {
            int rank = comm.Rank;
            int worldSize = comm.Size;
            
            int rowsPerProcess = n / worldSize;
            int remainder = n % worldSize;
            
            var localX = Vector<double>.Build.Dense(localL.RowCount);
            var globalX = Vector<double>.Build.Dense(n);
            
            for (int i = n - 1; i >= 0; i--)
            {
                int owner = FindProcessForRow(i, rowsPerProcess, remainder, worldSize);
                double xi = 0.0;
                
                if (rank == owner)
                {
                    int localI = GlobalToLocalIndex(i, rowsPerProcess, remainder, owner);
                    if (localI >= 0 && localI < localL.RowCount)
                    {
                        double sum = 0.0;
                        for (int j = i + 1; j < n; j++)
                        {
                            sum += localL[localI, j] * globalX[j];
                        }
                        
                        xi = (y[localI] - sum) / localL[localI, i];
                        globalX[i] = xi;
                    }
                }
                
                double xiToBroadcast = xi;
                comm.Broadcast(ref xiToBroadcast, owner);
                globalX[i] = xiToBroadcast;
                
                for (int loc = 0; loc < localL.RowCount; loc++)
                {
                    int globalIdx = LocalToGlobalIndex(loc, rank, rowsPerProcess, remainder);
                    if (globalIdx == i)
                    {
                        localX[loc] = xiToBroadcast;
                    }
                }
            }
            
            return localX;
        }
        
        private static int FindProcessForRow(int globalRow, int rowsPerProcess, int remainder, int worldSize)
        {
            for (int p = 0; p < worldSize; p++)
            {
                int start = p * rowsPerProcess + Math.Min(p, remainder);
                int end = start + rowsPerProcess + (p < remainder ? 1 : 0);
                
                if (globalRow >= start && globalRow < end)
                {
                    return p;
                }
            }
            return 0;
        }
        
        private static int GlobalToLocalIndex(int globalIndex, int rowsPerProcess, int remainder, int process)
        {
            int start = process * rowsPerProcess + Math.Min(process, remainder);
            return globalIndex - start;
        }
        
        private static int LocalToGlobalIndex(int localIndex, int process, int rowsPerProcess, int remainder)
        {
            int start = process * rowsPerProcess + Math.Min(process, remainder);
            return start + localIndex;
        }

        //some feature
    }
}