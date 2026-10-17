Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

pythonPath = "E:\ComfyUI\venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonPath) Then
    pythonPath = "E:\ComfyUI\venv\Scripts\python.exe"
End If
If Not fso.FileExists(pythonPath) Then
    pythonPath = "pythonw.exe"
End If

scriptPath = "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D\persistent_runner.py"
workingDir = "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D"

WshShell.CurrentDirectory = workingDir
WshShell.Run """" & pythonPath & """ """ & scriptPath & """", 0, False
