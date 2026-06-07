import subprocess, sys
result = subprocess.run([sys.executable, "-m", "pip", "install", "fastapi"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("CODE:", result.returncode)
