require('dotenv').config();
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');
const { exec, execSync } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');
const db = require('./database');
const { optimizeSensorPlacement } = require('./sensorPlacement');

const app = express();
const port = process.env.PORT || 3000;

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'your-placeholder-key',
});

// Load knowledge base for content lookup
let kb = [];
let projectIndex = {};
const kbPath = path.join(__dirname, 'rag_research.json');
const indexPath = path.join(__dirname, 'index.json');

try {
  if (fs.existsSync(kbPath)) {
    const ragData = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
    kb = ragData.knowledge_base || [];
  } else {
    console.warn('Warning: rag_research.json not found. Retrieval matching will fail.');
  }
  
  if (fs.existsSync(indexPath)) {
    projectIndex = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  }
} catch (err) {
  console.error('Error loading data files:', err.message);
}

app.use(helmet());
app.use(cors());
app.use(morgan('dev'));
app.use(express.json());
app.use(express.static('public'));

app.get('/api', (req, res) => {
  res.json({
    message: 'High-Precision Scientific Computing & RAG API',
    metadata: projectIndex.project_metadata || {},
    endpoints: [
      { path: '/simulations', method: 'GET', description: 'List available simulations' },
      { path: '/batch-run', method: 'POST', description: 'Run multiple simulations and get a comparative AI summary' },
      { path: '/chat', method: 'POST', description: 'Interact with the AI Research Assistant' },
      { path: '/retrieve?v=v1,v2,v3', method: 'GET', description: 'Retrieve context using vector search' }
    ]
  });
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/sensor-placement', (req, res) => {
  try {
    const result = optimizeSensorPlacement(req.body || {});
    res.json(result);
  } catch (error) {
    console.error('Sensor placement error:', error.message);
    res.status(500).json({
      error: 'Error optimizing sensor placement.',
      details: error.message,
    });
  }
});

// AI Research Assistant Endpoint with Function Calling
app.post('/chat', async (req, res) => {
  const { message, history } = req.body;
  if (!message) return res.status(400).json({ error: 'Message is required.' });

  const tools = [
    {
      type: "function",
      function: {
        name: "list_simulations",
        description: "List all available scientific simulations in the system.",
        parameters: { type: "object", properties: {} }
      }
    },
    {
      type: "function",
      function: {
        name: "run_simulation",
        description: "Execute a specific simulation and return its raw output.",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string", description: "The filename of the simulation (e.g., 'biot_savart_precision.exe')." }
          },
          required: ["name"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "retrieve_context",
        description: "Retrieve scientific context or development conventions from the knowledge base using a 10D vector.",
        parameters: {
          type: "object",
          properties: {
            vector: { type: "array", items: { type: "number" }, minItems: 10, maxItems: 10, description: "A 10-dimensional vector for similarity search." }
          },
          required: ["vector"]
        }
      }
    }
  ];

  try {
    let messages = [
      { role: "system", content: "You are a Senior AI Research Assistant. You help scientists run simulations, analyze results, and retrieve knowledge. Use the provided tools when necessary. Always explain the significance of simulation results." },
      ...(history || []),
      { role: "user", content: message }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: messages,
      tools: tools,
    });

    const assistantMessage = response.choices[0].message;

    if (assistantMessage.tool_calls) {
      for (const toolCall of assistantMessage.tool_calls) {
        const functionName = toolCall.function.name;
        const args = JSON.parse(toolCall.function.arguments);
        let functionResult;

        if (functionName === "list_simulations") {
          const binDir = path.join(__dirname, 'bin');
          const files = fs.readdirSync(binDir);
          functionResult = files.filter(f => f.endsWith('.exe'));
        } else if (functionName === "run_simulation") {
          let simName = args.name;
          if (process.platform === 'win32' && !simName.toLowerCase().endsWith('.exe')) simName += '.exe';
          const binPath = path.join(__dirname, 'bin', simName);
          if (fs.existsSync(binPath)) {
            const { stdout } = await execPromise(`"${binPath}"`);
            functionResult = stdout;
          } else {
            functionResult = "Error: Simulation not found.";
          }
        } else if (functionName === "retrieve_context") {
          const vec = args.vector.join(',');
          const binPath = path.join(__dirname, 'bin', process.platform === 'win32' ? 'rag_precision_retrieval.exe' : 'rag_precision_retrieval');
          const resultId = execSync(`"${binPath}" ${vec}`).toString().trim();
          const entry = kb.find(item => item.id == resultId);
          functionResult = entry ? entry.content : "No match found.";
        }

        messages.push(assistantMessage);
        messages.push({
          tool_call_id: toolCall.id,
          role: "tool",
          name: functionName,
          content: JSON.stringify(functionResult),
        });
      }

      const secondResponse = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: messages,
      });

      res.json({
        reply: secondResponse.choices[0].message.content,
        history: [...messages, secondResponse.choices[0].message]
      });
    } else {
      res.json({
        reply: assistantMessage.content,
        history: [...messages, assistantMessage]
      });
    }

  } catch (error) {
    console.error('Chat Error:', error.message);
    res.status(500).json({ error: 'Error in AI Research Assistant.', details: error.message });
  }
});

// Run multiple simulations and generate a consolidated AI summary
app.post('/batch-run', async (req, res) => {
  const { simulations } = req.body;
  if (!Array.isArray(simulations) || simulations.length === 0) {
    return res.status(400).json({ error: 'Please provide an array of simulation names.' });
  }

  const results = [];
  try {
    for (const sim of simulations) {
      let simName = sim;
      if (process.platform === 'win32' && !simName.toLowerCase().endsWith('.exe')) {
        simName += '.exe';
      }
      const binPath = path.join(__dirname, 'bin', simName);
      if (fs.existsSync(binPath)) {
        console.log(`Batch: Executing ${simName}`);
        const { stdout } = await execPromise(`"${binPath}"`);
        results.push({ name: simName, output: stdout });
      } else {
        results.push({ name: simName, error: 'Executable not found' });
      }
    }

    const summaryPrompt = results.map(r => `Simulation: ${r.name}\nOutput: ${r.output || r.error}`).join('\n\n---\n\n');

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { 
            role: "system", 
            content: "You are a senior research coordinator. You have just completed a batch of high-precision simulations. Provide a consolidated summary report that highlights the key findings from each simulation and discusses any interdisciplinary connections or comparative insights." 
          },
          { 
            role: "user", 
            content: summaryPrompt 
          }
        ],
      });

      res.json({
        batch_results: results,
        consolidated_summary: completion.choices[0].message.content
      });
    } catch (aiError) {
      console.error('Batch AI Error:', aiError.message);
      res.json({
        batch_results: results,
        consolidated_summary: "Consolidated AI Summary unavailable. Please check your OPENAI_API_KEY.",
        ai_error: aiError.message
      });
    }

  } catch (error) {
    console.error('Batch error:', error.message);
    res.status(500).json({ 
      error: 'Error during batch processing.',
      details: error.message,
      partial_results: results 
    });
  }
});

// Retrieve context using vector search (executes rag_precision_retrieval.exe)
app.get('/retrieve', (req, res) => {
  const queryParam = req.query.v;
  if (!queryParam) {
    return res.status(400).json({ error: 'Missing query vector. Example: /retrieve?v=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0' });
  }

  const vec = queryParam.split(',');
  if (vec.length !== 10) {
    return res.status(400).json({ error: 'Vector must have 10 components.' });
  }

  try {
    let binName = 'rag_precision_retrieval';
    if (process.platform === 'win32') binName += '.exe';
    const binPath = path.join(__dirname, 'bin', binName);
    
    if (!fs.existsSync(binPath)) {
      return res.status(500).json({ error: 'Retrieval engine binary not found. Please run build.' });
    }

    const cmd = `"${binPath}" ${vec.join(' ')}`;
    const resultId = execSync(cmd).toString().trim();
    
    const entry = kb.find(item => item.id == resultId);
    if (entry) {
      res.json({
        id: entry.id,
        content: entry.content,
        match_vector: entry.vector
      });
    } else {
      res.status(404).json({ error: 'No match found in knowledge base.' });
    }
  } catch (error) {
    console.error('Retrieval error:', error.message);
    res.status(500).json({ error: 'Error executing retrieval engine.' });
  }
});

// List all compiled simulations in the bin directory
app.get('/simulations', (req, res) => {
  const binDir = path.join(__dirname, 'bin');
  if (!fs.existsSync(binDir)) {
    return res.status(404).json({ error: 'Bin directory not found. Please run build.' });
  }
  fs.readdir(binDir, (err, files) => {
    if (err) return res.status(500).json({ error: 'Could not read bin directory' });
    const executables = files.filter(f => f.endsWith('.exe') || (!f.includes('.') && process.platform !== 'win32'));
    res.json({ simulations: executables });
  });
});

// Run a specific simulation
app.post('/run/:name', (req, res) => {
  let simName = req.params.name;
  if (process.platform === 'win32' && !simName.toLowerCase().endsWith('.exe')) {
    simName += '.exe';
  }
  const binPath = path.join(__dirname, 'bin', simName);

  if (!fs.existsSync(binPath)) {
    return res.status(404).json({ error: `Simulation ${simName} not found` });
  }

  console.log(`Executing simulation: ${simName}`);
  exec(`"${binPath}"`, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({
        error: error.message,
        stderr: stderr
      });
    }
    res.json({
      simulation: simName,
      output: stdout,
      error: stderr
    });
  });
});

// Run a specific simulation and analyze results with AI
app.post('/analyze/:name', (req, res) => {
  let simName = req.params.name;
  if (process.platform === 'win32' && !simName.toLowerCase().endsWith('.exe')) {
    simName += '.exe';
  }
  const binPath = path.join(__dirname, 'bin', simName);

  if (!fs.existsSync(binPath)) {
    return res.status(404).json({ error: `Simulation ${simName} not found` });
  }

  console.log(`Executing and analyzing simulation: ${simName}`);
  exec(`"${binPath}"`, async (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: error.message, stderr: stderr });
    }

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { 
            role: "system", 
            content: "You are a senior scientist specializing in high-precision numerical analysis and physics simulations. Explain the following simulation results in detail, focusing on scientific significance and potential implications." 
          },
          { 
            role: "user", 
            content: `Simulation: ${simName}\n\nOutput:\n${stdout}` 
          }
        ],
      });

      res.json({
        simulation: simName,
        raw_output: stdout,
        analysis: completion.choices[0].message.content
      });
    } catch (aiError) {
      console.error('AI Analysis Error:', aiError.message);
      res.json({
        simulation: simName,
        raw_output: stdout,
        analysis: "AI Analysis unavailable. Please check your OPENAI_API_KEY.",
        ai_error: aiError.message
      });
    }
  });
});

const server = app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});

server.on('error', (error) => {
  console.error('Server error:', error.message);
});

server.on('close', () => {
  console.warn('Server closed.');
});

module.exports = { app, server };
