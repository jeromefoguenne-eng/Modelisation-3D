Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

pythonwPath = "E:\ComfyUI\venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonwPath) Then
    pythonwPath = "pythonw.exe"
End If

scriptPath = "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D\persistent_runner.py"
workingDir = "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D"

WshShell.CurrentDirectory = workingDir
WshShell.Run """" & pythonwPath & """ """ & scriptPath & """", 0, False
