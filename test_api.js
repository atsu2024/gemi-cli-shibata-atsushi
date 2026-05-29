const axios = require('axios');

async function runTests() {
  const baseUrl = 'http://localhost:3000';
  let localServer;
  let failures = 0;

  const pass = (message) => console.log(`PASS ${message}`);
  const fail = (message, details) => {
    failures += 1;
    console.error(`FAIL ${message}`, details || '');
  };
  const request = async (config, retries = 2) => {
    try {
      return await axios(config);
    } catch (error) {
      const retryableCodes = ['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT'];
      if (retries > 0 && retryableCodes.includes(error.code)) {
        await new Promise((resolve) => setTimeout(resolve, 500));
        return request(config, retries - 1);
      }
      throw error;
    }
  };

  console.log('--- Starting API Tests ---');

  try {
    try {
      await axios.get(`${baseUrl}/api`, { timeout: 1000 });
    } catch (_) {
      localServer = require('./server').server;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    // Test Root (can be JSON or HTML Dashboard)
    console.log('Testing GET /');
    const rootRes = await request({ method: 'get', url: `${baseUrl}/` });
    const rootData = typeof rootRes.data === 'string' ? rootRes.data : rootRes.data.message;
    if (
      rootData &&
      (rootData.includes('Scientific Computing') ||
        rootData.includes('Break-Even Point Analysis') ||
        (rootData.includes('<div id="root"></div>') && rootData.includes('app.js')))
    ) {
      pass('GET /');
    } else {
      fail('GET /', rootData);
    }

    // Test /simulations
    console.log('Testing GET /simulations');
    const simRes = await request({ method: 'get', url: `${baseUrl}/simulations` });
    if (Array.isArray(simRes.data.simulations) && simRes.data.simulations.length > 0) {
      pass(`GET /simulations (${simRes.data.simulations.length} found)`);
    } else {
      fail('GET /simulations', simRes.data);
    }

    // Test /retrieve (High Precision C)
    console.log('Testing GET /retrieve (High Precision C)');
    const retrieveRes = await request({ method: 'get', url: `${baseUrl}/retrieve?v=0.20,1.00,0.05,0.25,0.10,1.00,0.05,0.55,0.25,0.05` });
    if (retrieveRes.data.id === 3) {
      pass('GET /retrieve (ID 3)');
    } else {
      fail('GET /retrieve (ID 3)', retrieveRes.data);
    }

    // Test /retrieve (Deep Learning)
    console.log('Testing GET /retrieve (Deep Learning)');
    const dlRes = await request({ method: 'get', url: `${baseUrl}/retrieve?v=0.4,0.4,0.8,0.5,0.6,0.1,0.2,0.3,0.4,0.5` });
    if (dlRes.data.id === 4) {
      pass('GET /retrieve (ID 4)');
    } else {
      fail('GET /retrieve (ID 4)', dlRes.data);
    }

    // Test /run/:name
    const simName = 'biot_savart_precision.exe';
    console.log(`Testing POST /run/${simName}`);
    const runRes = await request({ method: 'post', url: `${baseUrl}/run/${simName}` });
    if (runRes.data.output.includes('long double')) {
      pass(`POST /run/${simName}`);
    } else {
      fail(`POST /run/${simName}`, runRes.data);
    }

    // Test /analyze/:name
    console.log(`Testing POST /analyze/${simName}`);
    const analyzeRes = await request({ method: 'post', url: `${baseUrl}/analyze/${simName}` });
    if (analyzeRes.data.raw_output && analyzeRes.data.analysis) {
      pass(`POST /analyze/${simName} (Analysis: ${analyzeRes.data.analysis.substring(0, 30)}...)`);
    } else {
      fail(`POST /analyze/${simName}`, analyzeRes.data);
    }

    // Test /batch-run
    console.log('Testing POST /batch-run');
    const batchRes = await request({
      method: 'post',
      url: `${baseUrl}/batch-run`,
      data: {
        simulations: ['biot_savart_precision.exe', 'double_pendulum_precision.exe'],
      },
    });
    if (Array.isArray(batchRes.data.batch_results) && batchRes.data.batch_results.length === 2 && batchRes.data.consolidated_summary) {
      pass('POST /batch-run');
    } else {
      fail('POST /batch-run', batchRes.data);
    }
  } catch (error) {
    failures += 1;
    console.error('Test error detail:', error.message);
    if (error.response) {
      console.error('Status:', error.response.status);
      console.error('Data:', error.response.data);
    }
  }

  if (localServer) {
    await new Promise((resolve) => localServer.close(resolve));
  }

  console.log('--- Tests Completed ---');
  if (failures > 0) {
    console.error(`${failures} test(s) failed.`);
    process.exitCode = 1;
  }
}

runTests();
