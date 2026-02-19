const { STAGES } = require("../modules/sessionStore");

async function handleAskOptimize(session, message) {

  session.userResponses.optimize = message;
  session.stage = STAGES.ASK_USER_TESTCASE;

  return `
Now let's talk about optimization.

🧠 Explain:
- Time complexity of your solution
- Space complexity
- Any edge cases
- How would you improve it further if constraints increased?

After this, you will give your own test case.
`;
}

module.exports = { handleAskOptimize };
