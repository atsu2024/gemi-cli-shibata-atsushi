const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'research.db');
const db = new sqlite3.Database(dbPath);

db.serialize(() => {
  // Simulation History Table
  db.run(`CREATE TABLE IF NOT EXISTS simulation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    raw_output TEXT,
    ai_analysis TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )`);

  // Chat History Table
  db.run(`CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )`);
});

module.exports = {
  saveSimulation: (name, raw, ai) => {
    return new Promise((resolve, reject) => {
      db.run(
        `INSERT INTO simulation_runs (name, raw_output, ai_analysis) VALUES (?, ?, ?)`,
        [name, raw, ai],
        function(err) {
          if (err) reject(err);
          else resolve(this.lastID);
        }
      );
    });
  },
  saveChatMessage: (role, content) => {
    return new Promise((resolve, reject) => {
      db.run(
        `INSERT INTO chat_messages (role, content) VALUES (?, ?)`,
        [role, content],
        function(err) {
          if (err) reject(err);
          else resolve(this.lastID);
        }
      );
    });
  },
  getSimulationHistory: (limit = 10) => {
    return new Promise((resolve, reject) => {
      db.all(`SELECT * FROM simulation_runs ORDER BY timestamp DESC LIMIT ?`, [limit], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  },
  getChatHistory: (limit = 50) => {
    return new Promise((resolve, reject) => {
      db.all(`SELECT * FROM chat_messages ORDER BY timestamp ASC LIMIT ?`, [limit], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
};
