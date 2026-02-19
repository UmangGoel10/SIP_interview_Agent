const { pickRandomQuestion } = require("../modules/questions");
const { STAGES } = require("../modules/sessionStore");

async function handleShowQuestion(session) {

  // Assign question only once per session
  if (!session.problem) {
    session.problem = pickRandomQuestion();
  }

  const q = session.problem;

  session.stage = STAGES.ASK_APPROACH;

  return `
🧩 Coding Problem Selected

Topic: ${q.topic}
Subtopic: ${q.subtopic}
Difficulty: ${q.difficulty}

Problem: ${q.title}
LeetCode: ${q.link}

Now — explain your approach to solve this problem.
(Do NOT write code yet. Focus on logic first.)
  `;
}

module.exports = { handleShowQuestion };
