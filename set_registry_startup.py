import winreg

key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE)

# Clean path to batch file with quotes
target_cmd = '"C:\\Users\\TrendingPC\\.gemini\\antigravity\\scratch\\Modelisation-3D\\run_service.bat"'

# Remove any old broken keys
for old in ["Studio3D_Watchdog"]:
    try:
        winreg.DeleteValue(key, old)
        print(f"Removed old key: {old}")
    except FileNotFoundError:
        pass

winreg.SetValueEx(key, "Studio3D_Service", 0, winreg.REG_SZ, target_cmd)
saved_val, _ = winreg.QueryValueEx(key, "Studio3D_Service")
print("SUCCESSFULLY SET IN REGISTRY HKCU\\Run:")
print("Name : Studio3D_Service")
print("Value:", saved_val)
