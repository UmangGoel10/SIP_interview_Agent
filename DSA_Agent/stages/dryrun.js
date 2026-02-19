const { STAGES } = require("../modules/sessionStore");

async function handleAskDryRun(session, message) {

  session.userResponses.dryrun = message;

  session.stage = STAGES.ASK_OPTIMIZE;

  const q = session.problem;

  return `
Now do a dry-run of your code on the following sample input.

🔍 Problem:
${q.title}

✍️ Perform a step-by-step trace:
- Show variable values as they change
- Mention important decision points
- Explain how final output is reached

After dry-run — you will be asked to optimize the solution.
`;
}

module.exports = { handleAskDryRun };
