

import os
import subprocess
import sys

def print_header(title):
    print("\n" + "="*60)
    print(f"\t{title}")
    print("="*60)

def run_command(command, cwd='.', shell=True):
    print(f"\n> Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        process = subprocess.Popen(
            command, 
            cwd=cwd, 
            shell=shell, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
        
        for line in process.stderr:
            print(line, end='', file=sys.stderr)
            
        process.wait()
        if process.returncode != 0:
            print(f"\n[ERROR] Command failed with exit code {process.returncode}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(f"\n[ERROR] Command not found: {command[0] if isinstance(command, list) else command.split()[0]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        return False

def check_ollama():
    print_header("Checking for Ollama")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("Ollama is installed.")
            return True
    except FileNotFoundError:
        pass
    
    print("[WARNING] Ollama is not installed or not in the system's PATH.", file=sys.stderr)
    print("Please download and install Ollama from: https://ollama.com/")
    if sys.platform == "win32":
        print("After installation, ensure the Ollama directory is added to your PATH.")
    elif sys.platform == "darwin":
        print("Ollama will be installed in /usr/local/bin, which should be in your PATH.")
    else:
        print("Follow the Linux installation instructions on the Ollama website.")
    return False

def pull_llama3_model():
    print_header("Downloading Llama 3 Model")
    if not run_command("ollama pull llama3"):
        print("\n[ERROR] Failed to download the Llama 3 model.", file=sys.stderr)
        print("Please ensure Ollama is running and try again.", file=sys.stderr)
        return False
    return True

def install_dependencies():
    print_header("Installing Dependencies")
    
    # Root npm dependencies
    print("\n--- Installing root npm dependencies ---")
    if not run_command("npm install", cwd="."):
        return False
        
    # React app npm dependencies
    print("\n--- Installing React app dependencies ---")
    if not run_command("npm install", cwd="my-app"):
        return False
        
    # AI Doctor Python dependencies
    print("\n--- Installing AI Doctor Python dependencies ---")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", cwd="my-app/AI doctor"):
        return False
        
    # Analyzer Python dependencies
    print("\n--- Installing Analyzer Python dependencies ---")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", cwd="analzer"):
        return False
        
    return True

def verify_setup():
    print_header("Verifying Setup")
    required_files = [
        "my-app/start-servers.js",
        "my-app/server.js",
        "my-app/src/App.js",
        "my-app/AI doctor/gradio_app.py",
        "analzer/app.py"
    ]
    
    all_files_found = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"[ERROR] Missing required file: {file_path}", file=sys.stderr)
            all_files_found = False
            
    if all_files_found:
        print("\nAll required files are present.")
    else:
        print("\n[ERROR] Some required files are missing. Please check the repository.", file=sys.stderr)
        
    return all_files_found

def main():
    print_header("Medico AI Setup Script")
    
    if not check_ollama():
        sys.exit(1)
        
    if not pull_llama3_model():
        sys.exit(1)
        
    if not install_dependencies():
        sys.exit(1)
        
    if not verify_setup():
        sys.exit(1)
        
    print_header("Setup Complete!")
    print("\nYou can now run the application using: python run.py")

if __name__ == "__main__":
    main()
