const STAGES = {
  SHOW_QUESTION: "SHOW_QUESTION",
  ASK_APPROACH: "ASK_APPROACH",
  EVALUATE_APPROACH: "EVALUATE_APPROACH",
  ASK_PSEUDOCODE: "ASK_PSEUDOCODE",
  ASK_CODE: "ASK_CODE",
  ASK_DRYRUN: "ASK_DRYRUN",
  ASK_OPTIMIZE: "ASK_OPTIMIZE",
  ASK_USER_TESTCASE: "ASK_USER_TESTCASE",
  COMPLETE: "COMPLETE"
};

const sessions = {};

function getSession(userId) {
  if (!sessions[userId]) {
    sessions[userId] = {
      id: userId,
      stage: STAGES.SHOW_QUESTION,
      problem: null,
      userResponses: {},
      feedback: {}
    };
  }
  return sessions[userId];
}

module.exports = {
  STAGES,
  sessions,
  getSession
};
