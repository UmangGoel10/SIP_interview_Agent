const { STAGES } = require("../modules/sessionStore");

async function handleAskPseudocode(session, message) {

  session.userResponses.pseudocode = message;

  session.stage = STAGES.ASK_CODE;

  return `
Great — now convert your approach into pseudocode.

📝 Guidelines:
- Use clear step-by-step logic
- No programming syntax required
- Show major decisions & loops
- Mention data structures used

After pseudocode — we will move to actual code.
`;
}

module.exports = { handleAskPseudocode };
