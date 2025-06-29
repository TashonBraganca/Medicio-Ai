# Medico AI: Your Personal Health Assistant

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Ollama](https://img.shields.io/badge/Ollama-2391FF?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

Medico AI is a comprehensive health application designed to provide users with AI-powered tools for managing their health. From analyzing medical reports to getting answers to health-related questions, Medico AI aims to be a one-stop solution for accessible health information.

## Features

*   **AI Doctor:** A conversational AI that can:
    *   Listen to your health concerns via voice or text.
    *   Analyze your symptoms and provide a preliminary assessment.
    *   Analyze medical images, such as rashes or scans.
*   **Scan Analyzer:** Upload medical reports (images or PDFs) to have them analyzed for key information and insights.
*   **Health Library:** Browse a collection of articles on various health topics.
*   **Interactive Chat:** A user-friendly chat interface for interacting with the AI.

## Tech Stack

Medico AI is built with a microservices-style architecture, combining the power of Python for AI and Node.js for a robust backend.

*   **Frontend:**
    *   React.js
    *   React Router
    *   Axios
    *   Bootstrap
*   **Backend:**
    *   **AI Services:**
        *   Python
        *   Gradio (for the AI Doctor UI)
        *   Flask (for the Scan Analyzer)
        *   Ollama (for running local LLMs)
        *   gTTS & SpeechRecognition (for voice interaction)
    *   **Web Services:**
        *   Node.js with Express.js
        *   MongoDB (for the Health Library)
*   **DevOps & Tooling:**
    *   `concurrently` & `nodemon`
    *   `pip` & `npm`

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Node.js and npm
*   Python and pip
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/Hackthon.git
    cd Hackthon
    ```

2.  **Run the setup script:**
    This script will check for Ollama, download the required AI model, and install all necessary dependencies.
    ```sh
    python setup.py
    ```

3.  **Run the application:**
    This will start all the servers and open the application in your default browser.
    ```sh
    python run.py
    ```

## Screenshots

*(Add screenshots of your application here to showcase its features.)*

**Example:**

| AI Doctor Chat | Scan Analyzer |
| :---: | :---: |
| ![AI Doctor Chat](link_to_your_screenshot.png) | ![Scan Analyzer](link_to_your_screenshot.png) |

## Project Structure

The project is organized as a monorepo with the following structure:

```
/
├── my-app/                 # Main React application
│   ├── AI doctor/          # Gradio-based AI doctor service
│   ├── analzer/            # Flask-based scan analyzer service
│   ├── public/             # Static assets for the React app
│   ├── server/             # Express.js backend for the React app
│   └── src/                # React app source code
├── .gitignore
├── package.json
├── setup.py                # The main setup script
├── run.py                  # The main run script
└── README.md
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.