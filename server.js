require('dotenv').config();
const express = require('express');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.get('/', (req, res) => {
  res.json({
    message: 'Scientific Computing & Deep Learning API',
    endpoints: [
      { path: '/simulations', method: 'GET', description: 'List available simulations' },
      { path: '/run/:name', method: 'POST', description: 'Run a specific simulation' }
    ]
  });
});

// List all compiled simulations in the bin directory
app.get('/simulations', (req, res) => {
  const binDir = path.join(__dirname, 'bin');
  fs.readdir(binDir, (err, files) => {
    if (err) return res.status(500).json({ error: 'Could not read bin directory' });
    const executables = files.filter(f => f.endsWith('.exe'));
    res.json({ simulations: executables });
  });
});

// Run a specific simulation
app.post('/run/:name', (req, res) => {
  const simName = req.params.name;
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

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
