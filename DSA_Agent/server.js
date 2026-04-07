require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { v4: uuidv4 } = require("uuid");

const { loadQuestionsFromCSV } = require("./modules/questions");
const { getSession, sessions, STAGES } = require("./modules/sessionStore");

// Stage handlers
const { handleShowQuestion } = require("./stages/askQuestion");
const { handleAskApproach, handleEvaluateApproach } = require("./stages/approach");
const { handleAskPseudocode } = require("./stages/pseudocode");
const { handleAskCode } = require("./stages/code");
const { handleAskDryRun } = require("./stages/dryrun");
const { handleAskOptimize } = require("./stages/optimize");
const { handleUserTestcase } = require("./stages/userTestcase");

const app = express();
app.use(cors());
app.use(express.json());

// Load CSV once at startup
(async () => {
  await loadQuestionsFromCSV("./data/dsa.csv");
})();

// ── NEW: Start / reset a session ────────────────────────────────────────────
app.get("/start/:userId", async (req, res) => {
  const userId = req.params.userId;

  // Delete existing session so it starts fresh
  if (sessions[userId]) {
    delete sessions[userId];
  }

  const session = getSession(userId); // creates a fresh session
  const reply = await handleShowQuestion(session);

  res.json({ userId, stage: session.stage, reply });
});

// ── NEW: Get current session status ─────────────────────────────────────────
app.get("/status/:userId", (req, res) => {
  const userId = req.params.userId;
  const session = sessions[userId];

  if (!session) {
    return res.status(404).json({ error: "Session not found. Call /start/:userId first." });
  }

  res.json({ userId, stage: session.stage });
});

// ── Existing: Main chat endpoint ─────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const userId = req.body.userId || uuidv4();
  const message = req.body.message || "";

  const session = getSession(userId);
  let reply;

  switch (session.stage) {
    case STAGES.SHOW_QUESTION:
      reply = await handleShowQuestion(session);
      break;

    case STAGES.ASK_APPROACH:
      reply = await handleAskApproach(session);
      break;

    case STAGES.EVALUATE_APPROACH:
      reply = await handleEvaluateApproach(session, message);
      break;

    case STAGES.ASK_PSEUDOCODE:
      reply = await handleAskPseudocode(session, message);
      break;

    case STAGES.ASK_CODE:
      reply = await handleAskCode(session, message);
      break;

    case STAGES.ASK_DRYRUN:
      reply = await handleAskDryRun(session, message);
      break;

    case STAGES.ASK_OPTIMIZE:
      reply = await handleAskOptimize(session, message);
      break;

    case STAGES.ASK_USER_TESTCASE:
      reply = await handleUserTestcase(session, message);
      break;

    default:
      reply = "Interview already completed.";
  }

  res.json({
    userId,
    stage: session.stage,
    reply
  });
});

app.listen(3000, () => {
  console.log("🚀 AI Interview Bot running on port 3000");
});
