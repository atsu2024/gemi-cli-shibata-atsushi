const express = require('express');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Load knowledge base and metadata
let kb = [];
let projectIndex = {};
try {
  kb = JSON.parse(fs.readFileSync(path.join(__dirname, 'rag_data.json'), 'utf8'));
  projectIndex = JSON.parse(fs.readFileSync(path.join(__dirname, 'index.json'), 'utf8'));
} catch (e) {
  console.warn('Warning: Could not load rag_data.json or index.json');
}

app.use(express.json());

app.get('/', (req, res) => {
  res.json({
    status: 'High-Precision Scientific RAG Bridge is active.',
    metadata: projectIndex.project_metadata || {},
    usage: 'Use /retrieve?v=v1,v2,v3,v4,v5 to query.'
  });
});

app.get('/retrieve', (req, res) => {
  const queryParam = req.query.v;
  if (!queryParam) {
    return res.status(400).send('Missing query vector. Example: /retrieve?v=0.1,0.2,0.3,0.4,0.5');
  }

  const vec = queryParam.split(',');
  if (vec.length !== 5) {
    return res.status(400).send('Vector must have 5 components.');
  }

  try {
    // Execute the C program with high-precision arguments
    const binName = process.platform === 'win32' ? 'rag_precision_retrieval.exe' : 'rag_precision_retrieval';
    const cmd = `"${path.join(__dirname, 'bin', binName)}" ${vec.join(' ')}`;
    const resultId = execSync(cmd).toString().trim();
    
    const entry = kb.find(item => item.id == resultId);
    if (entry) {
      res.json({
        id: entry.id,
        content: entry.content,
        match_vector: entry.vector
      });
    } else {
      res.status(404).send('No match found.');
    }
  } catch (error) {
    console.error(error);
    res.status(500).send('Error executing retrieval engine.');
  }
});

app.listen(port, () => {
  console.log(`RAG Bridge running on http://localhost:${port}`);
});
