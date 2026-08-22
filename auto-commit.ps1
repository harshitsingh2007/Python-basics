$project = "C:\Users\HARSHIT SINGH\OneDrive\Apps\Desktop\pythonFile"

Set-Location $project

Get-Date | Out-File "$project\last-updated.txt"

git add .

git commit -m "chore: daily update"

git push