# SIP Interview Agent

An AI-powered interview system that conducts structured technical interviews based on candidate resumes. The agent analyzes project experience and conducts deep-dive interviews, with optional DSA (Data Structures & Algorithms) coding rounds.

## Features

- **Resume Analysis**: Automatically extracts projects and technical stack from PDF resumes
- **Structured Interview Flow**: Multi-stage interview with question generation, answer analysis, and depth probing
- **Adaptive Questioning**: Dynamically generates follow-up questions based on candidate responses and coverage gaps
- **Project-Based Interview**: Interviews multiple projects with complexity-based sorting
- **Optional DSA Round**: Seamlessly integrates with DSA coding challenges after resume interview
- **Configurable Behavior**: Control via CLI args or environment variables for production deployments

## Architecture

The system uses **LangGraph** for state management and interview flow orchestration:

```
Resume PDF
    ↓
[PDFConverter] → Markdown
    ↓
[ProjectExtractor] → List[Project]
    ↓
[Interview Graph]
    ├→ node1_question_generator: Generate questions based on intent
    ├→ node2_answer_input: Collect candidate answer
    ├→ node3_answer_analysis: LLM-driven analysis + decision routing
    ├→ Conditional routing (FollowUp, SwitchProject, MoveOn, WrapUp)
    ├→ node4_final_performance_analysis: Summary & recommendations
    ↓
[Optional DSA Round]
```

## Installation

### Prerequisites

- Python 3.12+
- pip or uv package manager
- BAML client configured with LLM credentials

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/SIP_Interview_Agent
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   uv pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIP_RESUME_FILE` | (auto-detect) | Path to resume PDF file |
| `SIP_DSA_MODE` | `prompt` | DSA round behavior: `prompt`, `always`, or `never` |

Other variables (e.g., API keys) should be configured in `.env`.

### Runtime Defaults

- **Max Interview Turns**: 20
- **Max Follow-ups per Question**: 3
- **Max Questions per Project**: 6

## Usage

### Basic Usage (Interactive)

Run with auto-detection of resume PDF in current directory:
```bash
python arch.py
```

### Specify Resume File

```bash
# Using CLI argument
python arch.py --resume-file /absolute/path/to/resume.pdf

# Using environment variable
export SIP_RESUME_FILE=/absolute/path/to/resume.pdf
python arch.py
```

### Control DSA Round

```bash
# Always run DSA round (no prompt)
python arch.py --resume-file resume.pdf --dsa-mode always

# Never run DSA round
python arch.py --resume-file resume.pdf --dsa-mode never

# Prompt user (default)
python arch.py --resume-file resume.pdf --dsa-mode prompt
```

### Non-Interactive / CI/CD

```bash
# Disable DSA prompt for automated runs
python arch.py --resume-file resume.pdf --dsa-mode never

# Or set environment variable
export SIP_DSA_MODE=never
python arch.py --resume-file resume.pdf
```

## CLI Reference

```bash
python arch.py --help
```

### Options

- `--resume-file PATH`: Path to resume PDF (optional)
  - If omitted, reads `$SIP_RESUME_FILE` environment variable
  - If not set, auto-detects single PDF in current directory
  - Error if zero or multiple PDFs found without explicit selection

- `--dsa-mode {prompt,always,never}`: Control DSA coding round (default: `prompt`)
  - `prompt`: Ask user after resume interview
  - `always`: Automatically start DSA round
  - `never`: Skip DSA round entirely
  - Can be overridden by `$SIP_DSA_MODE` env var

## Project Structure

```
SIP_Interview_Agent/
├── arch.py                 # Main interview orchestration and CLI entry point
├── dsa_client.py          # Client for optional DSA coding round
├── resumeMD.py            # Resume PDF → Markdown converter
├── baml_client/           # BAML LLM client (auto-generated)
├── baml_src/              # BAML prompt/function definitions
│   ├── clients.baml
│   ├── generators.baml
│   └── prompts.baml
├── .env.example           # Example environment configuration
├── .venv/                 # Python virtual environment
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Interview Flow

1. **Resume Parsing**: PDF → Projects and tech stacks
2. **Project Sorting**: Sort by complexity (descending)
3. **Interview Loop** (per project):
   - Generate context-aware question
   - Collect candidate answer
   - Analyze answer (clarity, correctness, depth, coverage)
   - Route: FollowUp → more questions on same topic, or MoveOn → next project
   - Track follow-up count per question to avoid repetition
4. **Project Transitions**: Move to next uncovered project or finalize
5. **Final Analysis**: Summarize weaknesses, strengths, unanswered concepts
6. **Optional DSA Round**: Conduct separate coding challenges

## Example Run

```bash
# Terminal 1: Start DSA bot (if using DSA mode)
cd ../DSA_Agent
npm start
# Should output: 🚀 AI Interview Bot running on port 3000

# Terminal 2: Run interview
cd ../SIP_Interview_Agent
source .venv/bin/activate
python arch.py --resume-file ~/resumes/candidate.pdf --dsa-mode prompt
```

**Output:**
```
INFO: Using resume: /Users/example/resumes/candidate.pdf
Markdown Conversion Successful:
PROJECT 1: Machine Learning Pipeline
  - Topics covered: ML, Python, Data Engineering
  - Complexity: 9/10

Q: Can you explain the architecture and data flow in your ML pipeline?
A: [Candidate responds...]

[Interview continues with follow-ups and project transitions]

===== RESUME INTERVIEW COMPLETE =====

Would you like to continue with a DSA coding round? (y/n): y

🧩  DSA INTERVIEW BOT
[DSA interview begins...]
```

## Logging

Logs are printed to console with format: `LEVEL: MESSAGE`

Example:
```
INFO: Using resume: /path/to/resume.pdf
INFO: Parsing projects...
INFO: No projects found in resume.
ERROR: Resume analysis failed.
```

## Error Handling

The system provides clear error messages for common issues:

- **File not found**: "Resume file not found: /path/to/file"
- **Wrong file type**: "Resume file must be a PDF: /path/to/file.txt"
- **Multiple PDFs**: "Multiple PDFs found (a.pdf, b.pdf). Provide --resume-file..."
- **No PDFs found**: "No PDF found in /current/dir. Provide --resume-file or set SIP_RESUME_FILE..."
- **Conversion failed**: "Markdown conversion failed for /path/to/resume.pdf"
- **No projects extracted**: "No projects found in resume."

## Dependencies

Key dependencies (see `pyproject.toml`):
- `langgraph>=1.0.5` — Interview state machine and graph orchestration
- `baml-py==0.219.0` — LLM client and prompt management
- `pydantic>=2.12.5` — Data validation
- `python-dotenv>=1.2.1` — Environment variable management
- `docling>=2.66.0` — PDF processing
- `pandas>=2.3.3` — Data manipulation

## API Keys & Credentials

Ensure the following are configured in `.env`:

```bash
# Example .env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=...  # If using Groq for DSA bot
```

See `.env.example` for the complete template.

## Troubleshooting

### Resume not detected
```bash
# Check current directory
ls *.pdf

# Explicitly specify
python arch.py --resume-file /full/path/to/resume.pdf
```

### DSA bot connection error
Ensure DSA bot is running on port 3000:
```bash
cd ../DSA_Agent
npm start
```

### LLM API errors
Check `.env` file has valid API keys and network connectivity.

### Python version issues
Requires Python 3.12+:
```bash
python --version
python3.12 -m venv .venv
```

## Future Enhancements

- SQL database to persist interview sessions and results
- Async interview sessions (multiple candidates in parallel)
- Custom interview templates and scoring rubrics
- Export results to PDF reports
- Integration with recruiting platforms

## License

ISC

## Contributing

For bug reports and feature requests, please open an issue.
