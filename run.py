

import subprocess
import webbrowser
import time
import platform
import os

def print_header(title):
    print("\n" + "="*60)
    print(f"\t{title}")
    print("="*60)

def main():
    print_header("Running Medico AI")

    # Command to start all servers
    command = ["node", "start-servers.js"]
    
    # Determine the shell setting based on the OS
    shell = True if platform.system() == "Windows" else False

    try:
        # Start all the servers in the background
        print("\n> Starting all servers...")
        server_process = subprocess.Popen(command, cwd="my-app", shell=shell)

        # Give the servers a moment to start up
        print("\n> Waiting for servers to initialize...")
        time.sleep(15)  # Increased wait time for slower systems

        # Open the browser to the React app
        react_app_url = "http://localhost:3000"
        print(f"\n> Opening browser to: {react_app_url}")
        webbrowser.open(react_app_url)

        print("\n" + "-"*60)
        print("The application is now running. Press Ctrl+C to stop all servers.")
        print("-"*60)

        # Wait for the process to terminate
        server_process.wait()

    except FileNotFoundError:
        print("\n[ERROR] 'node' command not found. Please ensure Node.js is installed and in your PATH.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n\n> Shutting down all servers...")
        if server_process.poll() is None: # Check if the process is still running
            server_process.terminate()
            server_process.wait()
        print("\n> Servers have been stopped.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()

