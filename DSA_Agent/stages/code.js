const { STAGES } = require("../modules/sessionStore");

async function handleAskCode(session, message) {

  session.userResponses.code = message;

  session.stage = STAGES.ASK_DRYRUN;

  return `
Now write the full working code for your solution.

📝 Guidelines:
- Use any programming language
- Code should be clean & readable
- Variable names should be meaningful

After submitting code — we will do a dry run.
`;
}

module.exports = { handleAskCode };
