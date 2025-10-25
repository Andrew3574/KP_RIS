# final_tests.ps1
Write-Host "=== ТЕСТИРОВАНИЕ РАСПРЕДЕЛЕННОГО РЕШАТЕЛЯ СЛАУ ==="

Write-Host "`n1. Тест малой системы 50x50"
mpiexec -n 2 dotnet run -- test 50

Write-Host "`n2. Тест средней системы 200x200"
mpiexec -n 2 dotnet run -- test 200

Write-Host "`n3. Тест большой системы 500x500"
mpiexec -n 4 dotnet run -- test 500

Write-Host "`n4. Тест большой системы 1000x1000"
mpiexec -n 4 dotnet run -- test 1000

Write-Host "`n5. Тест с разным количеством процессов"
mpiexec -n 2 dotnet run -- test 300
mpiexec -n 4 dotnet run -- test 300
mpiexec -n 8 dotnet run -- test 300

Write-Host "`nВсе тесты завершены!"