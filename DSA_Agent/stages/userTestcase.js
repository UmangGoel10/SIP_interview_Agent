const { STAGES } = require("../modules/sessionStore");

async function handleUserTestcase(session, message) {

  session.userResponses.userTestcase = message;
  session.stage = STAGES.COMPLETE;

  return `
✅ Interview Complete!

Thanks for completing the interview round.

📝 Summary:
- Approach explained
- Pseudocode written
- Code implemented
- Dry-run performed
- Optimization discussed
- Custom test case provided

You can now start a new interview session if needed.
`;
}

module.exports = { handleUserTestcase };
