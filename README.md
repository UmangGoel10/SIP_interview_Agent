# Projects Overview

This repository contains two related interview systems:

- `DSA_Agent/`: a Node.js service that runs the DSA interview bot over HTTP
- `SIP_Interview_Agent/`: a Python interview orchestrator that reads a resume PDF, drives the technical interview flow, and can optionally hand off to the DSA bot

Together they form a resume-driven interview workflow with structured technical questioning and a separate coding round.

## Repository Layout

```text
Projects/
├── DSA_Agent/
│   ├── server.js
│   ├── package.json
│   ├── data/
│   └── stages/
├── SIP_Interview_Agent/
│   ├── arch.py
│   ├── dsa_client.py
│   ├── pyproject.toml
│   ├── resumeMD.py
│   └── baml_src/
└── README.md
```

## DSA_Agent

The DSA agent is an Express service that loads interview questions from CSV and moves a candidate through the coding interview stages.

### Features

- REST endpoints for starting and checking a session
- Stage-based interview flow
- CSV-backed question loading
- CORS and JSON middleware built in

### Setup

```bash
cd DSA_Agent
npm install
```

### Run

```bash
npm start
```

The server listens on port `3000`.

### Main Endpoints

- `GET /start/:userId` - starts or resets a session
- `GET /status/:userId` - returns the current session stage
- `POST /chat` - sends a message and receives the next interview prompt or response

## SIP_Interview_Agent

The SIP interview agent is a Python application that extracts project information from a resume PDF, generates technical interview questions with LangGraph, and can optionally continue into a DSA round.

### Features

- Resume PDF to Markdown conversion
- Project extraction and interview orchestration
- Configurable resume path via CLI or environment variable
- Optional DSA round with prompt, always, or never modes
- Structured logging and clear error handling

### Requirements

- Python 3.12+
- A virtual environment for the Python dependencies
- LLM credentials configured in `.env`

For the full SIP setup and runtime options, see [SIP_Interview_Agent/README.md](SIP_Interview_Agent/README.md).

### Setup

Create and activate a virtual environment, then install the dependencies declared in `pyproject.toml` with the Python package manager you use for this workspace.

### Run

```bash
python arch.py
```

You can also pass a specific resume file:

```bash
python arch.py --resume-file /absolute/path/to/resume.pdf
```

### Configuration

- `SIP_RESUME_FILE`: explicit path to the resume PDF
- `SIP_DSA_MODE`: controls the DSA round behavior (`prompt`, `always`, or `never`)

If no resume file is provided, the agent attempts to auto-detect a single PDF in the current directory.

## End-to-End Workflow

1. Start the DSA bot if you want the DSA round available.
2. Run the SIP interview agent with a resume PDF.
3. Complete the resume-based interview.
4. Choose whether to continue with the DSA round, or skip it with `--dsa-mode never`.

### Example

```bash
# Terminal 1
cd DSA_Agent
npm start

# Terminal 2
cd SIP_Interview_Agent
source .venv/bin/activate
python arch.py --resume-file ~/resumes/candidate.pdf --dsa-mode prompt
```

## Notes

- Each subproject also contains its own documentation.
- Keep secrets in local `.env` files and do not commit them.
- Generated artifacts such as `node_modules/`, `.venv/`, and `baml_client/` are ignored from version control.
