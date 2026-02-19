const { STAGES } = require("../modules/sessionStore");
const { askLLM } = require("../config/openai");

async function handleAskApproach(session) {
  session.stage = STAGES.EVALUATE_APPROACH;

  return `
Explain your approach to solve this problem.

📝 Guidelines:
- Do NOT write code yet
- Focus on logic and reasoning
- Mention time & space complexity (if you can)
`;
}

async function handleEvaluateApproach(session, message) {

  session.userResponses.approach = message;

  const q = session.problem;

  const feedback = await askLLM(
`You are an interview evaluator.

Evaluate whether the candidate's approach for this coding problem is CORRECT or WRONG.

Problem:
${q.title}

Topic: ${q.topic} — ${q.subtopic}
Difficulty: ${q.difficulty}

If the approach is WRONG:
- Identify the logical flaw
- Generate a counter example where it fails
- Explain why it fails
- Give a small hint (not full solution)

Respond in format:

WRONG:
<issue + counter example + hint>

If the approach is CORRECT:
Respond in format:

CORRECT:
<short validation + what they did right>
`, 
message
  );

  session.feedback.approach = feedback;

  // If wrong -> keep user in same stage to retry
  if (feedback.startsWith("WRONG")) {
    return `
❌ Your approach seems incorrect.

${feedback}

Try again — refine your logic based on the hint.
`;
  }
  session.stage = STAGES.ASK_PSEUDOCODE;

  return `
✅ Good — your approach is logically correct.

${feedback}

Now write pseudocode for your solution.
`;
}

module.exports = {
  handleAskApproach,
  handleEvaluateApproach
};
