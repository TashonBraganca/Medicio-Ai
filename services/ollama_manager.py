#!/usr/bin/env python3
"""
PERSISTENT OLLAMA SERVER MANAGER
Manages Ollama as a persistent background service for fast responses
"""

import subprocess
import time
import threading
import atexit
import signal
import os
import platform
import requests
import logging
from typing import Optional, Tuple
from config import config

logger = logging.getLogger(__name__)

class PersistentOllamaManager:
    """Manages Ollama as a persistent background service"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.startup_complete = False
        self.health_check_thread = None
        self.stop_requested = False

        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down Ollama...")
        self.cleanup()

    def start_persistent_server(self) -> Tuple[bool, str]:
        """Start Ollama as a persistent background service"""

        # Check if already running
        if self.is_server_running():
            logger.info("Ollama already running - using existing instance")
            self.is_running = True
            self.startup_complete = True
            return True, "Ollama already running"

        # Check if we already started it
        if self.is_running and self.process and self.process.poll() is None:
            return True, "Ollama already started by this manager"

        try:
            logger.info("Starting persistent Ollama server...")
            system_os = platform.system()

            # Start Ollama process based on OS
            if system_os == 'Windows':
                self.process = subprocess.Popen(
                    ['ollama', 'serve'],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            elif system_os == 'Darwin':  # macOS
                env = os.environ.copy()
                env['OLLAMA_HOST'] = '0.0.0.0:11434'
                self.process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
            else:  # Linux
                self.process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            logger.info(f"Ollama process started with PID: {self.process.pid}")

            # Wait for server to be ready
            startup_success = self._wait_for_startup()

            if startup_success:
                self.is_running = True
                self.startup_complete = True
                self._start_health_monitoring()
                logger.info("✅ Persistent Ollama server ready for fast responses!")
                return True, f"Ollama server started successfully (PID: {self.process.pid})"
            else:
                logger.error("Ollama server failed to start properly")
                return False, "Failed to start Ollama server"

        except Exception as e:
            logger.error(f"Error starting persistent Ollama server: {str(e)}")
            return False, f"Error starting Ollama: {str(e)}"

    def _wait_for_startup(self, max_wait: int = 30) -> bool:
        """Wait for Ollama server to be ready"""
        logger.info("Waiting for Ollama server to be ready...")

        for i in range(max_wait):
            if self.stop_requested:
                return False

            try:
                response = requests.get(
                    f"{config.OLLAMA_BASE_URL}/api/tags",
                    timeout=2
                )
                if response.status_code == 200:
                    logger.info(f"✅ Ollama server ready after {i+1} seconds")
                    return True
            except:
                pass

            # Check if process died
            if self.process and self.process.poll() is not None:
                logger.error("Ollama process died during startup")
                return False

            time.sleep(1)
            if i % 5 == 0:
                logger.info(f"Still waiting for Ollama... ({i+1}/{max_wait}s)")

        logger.error(f"Ollama server not ready after {max_wait} seconds")
        return False

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return

        self.health_check_thread = threading.Thread(
            target=self._health_monitor,
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Health monitoring started")

    def _health_monitor(self):
        """Monitor Ollama server health"""
        while self.is_running and not self.stop_requested:
            try:
                time.sleep(10)  # Check every 10 seconds

                if self.stop_requested:
                    break

                # Check if process is still alive
                if self.process and self.process.poll() is not None:
                    logger.warning("Ollama process died - attempting restart")
                    self.is_running = False
                    self.startup_complete = False
                    # Attempt restart
                    if not self.stop_requested:
                        self.start_persistent_server()
                    break

                # Check if server is responsive
                if not self.is_server_running():
                    logger.warning("Ollama server not responsive - checking...")
                    time.sleep(5)
                    if not self.is_server_running():
                        logger.error("Ollama server unresponsive - attempting restart")
                        self.cleanup()
                        if not self.stop_requested:
                            time.sleep(2)
                            self.start_persistent_server()
                        break

            except Exception as e:
                logger.error(f"Health monitor error: {str(e)}")
                time.sleep(5)

    def is_server_running(self) -> bool:
        """Check if Ollama server is running and responsive"""
        try:
            response = requests.get(
                f"{config.OLLAMA_BASE_URL}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False

    def wait_for_ready(self, max_wait: int = 30) -> bool:
        """Wait for the server to be ready for requests"""
        if self.startup_complete:
            return True

        for i in range(max_wait):
            if self.startup_complete:
                return True
            time.sleep(0.5)

        return self.startup_complete

    def get_server_status(self) -> dict:
        """Get current server status"""
        return {
            "manager_running": self.is_running,
            "startup_complete": self.startup_complete,
            "server_responsive": self.is_server_running(),
            "process_alive": self.process and self.process.poll() is None if self.process else False,
            "process_pid": self.process.pid if self.process else None
        }

    def cleanup(self):
        """Clean shutdown of Ollama server"""
        if self.stop_requested:
            return

        self.stop_requested = True
        logger.info("Shutting down persistent Ollama server...")

        # Stop health monitoring
        self.is_running = False

        # Terminate Ollama process
        if self.process:
            try:
                if platform.system() == 'Windows':
                    self.process.terminate()
                    time.sleep(2)
                    if self.process.poll() is None:
                        self.process.kill()
                else:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()

                logger.info(f"Ollama process {self.process.pid} terminated")
            except Exception as e:
                logger.error(f"Error terminating Ollama process: {str(e)}")

        self.process = None
        self.startup_complete = False

# Global instance
ollama_manager = PersistentOllamaManager()