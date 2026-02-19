const fs = require("fs");
const csv = require("csv-parser");

const questions = [];

function loadQuestionsFromCSV(path) {
  return new Promise((resolve, reject) => {
    let rowIndex = 0;

    fs.createReadStream(path)
      .pipe(csv())
      .on("data", (row) => {
        rowIndex++;

        const topic = row["Topic Name"]?.trim();
        const subtopic = row["Sub-topic Name"]?.trim();
        const difficulty = row["Difficulty"]?.trim();
        const title = row["LeetCode Question"]?.trim();
        const link = row["LeetCode Link"]?.trim();

        // Skip rows without a question
        if (!title) return;

        questions.push({
          id: `${rowIndex}-${title}`,
          topic,
          subtopic,
          difficulty,
          title,
          link,
          description: `${title} — ${topic} / ${subtopic}`
        });
      })
      .on("end", () => {
        console.log(`Loaded ${questions.length} questions from CSV`);
        resolve(questions);
      })
      .on("error", reject);
  });
}

function pickRandomQuestion() {
  return questions[Math.floor(Math.random() * questions.length)];
}

module.exports = {
  questions,
  loadQuestionsFromCSV,
  pickRandomQuestion
};
