const OpenAI = require("openai");

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

async function main() {
  try {
    const response = await client.chat.completions.create({
      model: "anthropic/claude-3-haiku",
      messages: [
        { role: "user", content: "C言語でPID制御コードを書いて" }
      ],
    });

    console.log(response.choices[0].message.content);

  } catch (err) {
    console.error("エラー:", err);
  }
}

main();