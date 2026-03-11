"""Git commit and push script"""
import subprocess
import os

os.chdir('G:\\Zed\\spatial mapping\\python')

# Commit
result = subprocess.run(['git', 'commit', '-m', 'feat: add crack segmentation scripts'], 
                       capture_output=True, text=True, encoding='utf-8')
print('Commit stdout:', result.stdout)
print('Commit stderr:', result.stderr)
print('Commit return code:', result.returncode)

# Push
result = subprocess.run(['git', 'push', 'origin', 'master'], 
                       capture_output=True, text=True, encoding='utf-8')
print('Push stdout:', result.stdout)
print('Push stderr:', result.stderr)
print('Push return code:', result.returncode)

print('Done!')
