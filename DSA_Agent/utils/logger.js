function logSessionEvent(session, message) {
  const ts = new Date().toISOString();

  console.log(`
-----------------------------
🧠 Interview Session Log
Time: ${ts}
User: ${session.id}
Stage: ${session.stage}
Problem: ${session?.problem?.title || "Not assigned"}
Event: ${message}
-----------------------------
`);
}

module.exports = { logSessionEvent };
