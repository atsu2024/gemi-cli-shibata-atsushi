const OpenAI = require("openai");
const readline = require("readline");

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function chat() {
  rl.question(">> ", async (input) => {
    try {
      const res = await client.chat.completions.create({
        model: "anthropic/claude-3-haiku",
        messages: [{ role: "user", content: input }],
      });

      console.log(res.choices[0].message.content);
    } catch (e) {
      console.log("エラー:", e.message);
    }

    chat();
  });
}

chat();