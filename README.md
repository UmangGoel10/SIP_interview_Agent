# AI Interview System

A two-part AI interviewing system for end-to-end technical interview practice:

1. **SIP_Interview_Agent** (Python) — Deep-dives into your resume projects using an AI agent (LangGraph + BAML)
2. **DSA_Agent** (Node.js) — Runs a structured DSA/LeetCode-style coding interview

---

## Setup

### 1. DSA_Agent (Node.js)

```bash
cd DSA_Agent
npm install
```

Create `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Start the server:
```bash
node server.js
# → 🚀 AI Interview Bot running on port 3000
```

### 2. SIP_Interview_Agent (Python)

```bash
cd SIP_Interview_Agent
```

Create `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Install dependencies:
```bash
uv sync
```

---

## Running

### Full Interview (Resume → DSA)

> Make sure `node server.js` is running first in `DSA_Agent/`

```bash
cd SIP_Interview_Agent
uv run python arch.py
```

Flow:
1. Reads your resume PDF and extracts projects
2. Conducts a project deep-dive interview (LangGraph agent)
3. Asks if you want a DSA round — if yes, calls `DSA_Agent` over HTTP automatically

### DSA Interview Only (from Python)

```bash
cd SIP_Interview_Agent
uv run python dsa_client.py
```

### DSA Interview via API directly

```bash
# Start a session
curl http://localhost:3000/start/my-session

# Send a message
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"userId": "my-session", "message": "I would use a hashmap..."}'

# Check current stage
curl http://localhost:3000/status/my-session
```

---

## DSA Interview Stages

```
Show Question → Approach → Evaluate → Pseudocode → Code → Dry Run → Optimize → Test Case → Complete
```

---

## Project Structure

```
Projects/
├── README.md
│
├── SIP_Interview_Agent/        # Python — resume/project interview bot
│   ├── arch.py                 # Main LangGraph interview agent
│   ├── dsa_client.py           # HTTP client → calls DSA_Agent API
│   ├── resumeMD.py             # Resume PDF → Markdown converter
│   ├── baml_src/               # BAML prompt definitions
│   ├── baml_client/            # Generated BAML client
│   ├── Primary_revised2.pdf    # Your resume (replace as needed)
│   └── .env                    # GROQ_API_KEY
│
└── DSA_Agent/                  # Node.js — DSA coding interview bot
    ├── server.js               # Express API server
    ├── config/openai.js        # Groq LLM client (OpenAI-compatible)
    ├── modules/                # Session store, CSV question loader
    ├── stages/                 # Per-stage interview handlers
    ├── data/dsa.csv            # LeetCode question bank
    └── .env                    # GROQ_API_KEY
```
