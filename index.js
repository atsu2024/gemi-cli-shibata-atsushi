const express = require('express');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Load knowledge base for content lookup
const kb = JSON.parse(fs.readFileSync(path.join(__dirname, 'rag_data.json'), 'utf8'));

app.use(express.json());

app.get('/', (req, res) => {
  res.send('High-Precision Scientific RAG Bridge is active. Use /retrieve?v=v1,v2,v3 to query.');
});

app.get('/retrieve', (req, res) => {
  const queryParam = req.query.v;
  if (!queryParam) {
    return res.status(400).send('Missing query vector. Example: /retrieve?v=0.1,0.9,0.5');
  }

  const vec = queryParam.split(',');
  if (vec.length !== 3) {
    return res.status(400).send('Vector must have 3 components.');
  }

  try {
    // Execute the C program with high-precision arguments
    const cmd = `${path.join(__dirname, 'bin', 'rag_retrieval.exe')} ${vec[0]} ${vec[1]} ${vec[2]}`;
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
