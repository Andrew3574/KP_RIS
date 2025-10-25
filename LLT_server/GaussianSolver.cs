// GaussianSolver.cs
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DistributedLLTSolver
{
    public static class GaussianSolver
    {
        public static Vector<double> Solve(Matrix<double> A, Vector<double> b)
        {
            int n = A.RowCount;
            var Ab = A.Clone(); // Расширенная матрица
            var x = Vector<double>.Build.Dense(n);
            
            // Прямой ход
            for (int i = 0; i < n; i++)
            {
                // Поиск главного элемента
                int maxRow = i;
                double maxVal = Math.Abs(Ab[i, i]);
                
                for (int k = i + 1; k < n; k++)
                {
                    if (Math.Abs(Ab[k, i]) > maxVal)
                    {
                        maxVal = Math.Abs(Ab[k, i]);
                        maxRow = k;
                    }
                }
                
                // Перестановка строк
                if (maxRow != i)
                {
                    for (int j = i; j < n; j++)
                    {
                        double temp = Ab[i, j];
                        Ab[i, j] = Ab[maxRow, j];
                        Ab[maxRow, j] = temp;
                    }
                    double tempB = b[i];
                    b[i] = b[maxRow];
                    b[maxRow] = tempB;
                }
                
                // Исключение
                for (int k = i + 1; k < n; k++)
                {
                    double factor = Ab[k, i] / Ab[i, i];
                    for (int j = i; j < n; j++)
                    {
                        Ab[k, j] -= factor * Ab[i, j];
                    }
                    b[k] -= factor * b[i];
                }
            }
            
            // Обратный ход
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0.0;
                for (int j = i + 1; j < n; j++)
                {
                    sum += Ab[i, j] * x[j];
                }
                x[i] = (b[i] - sum) / Ab[i, i];
            }
            
            return x;
        }
    }
}