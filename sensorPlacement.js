function parsePositiveNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function round(value, digits = 3) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function buildGrid(width, height, spacing) {
  const points = [];
  for (let y = 0; y <= height + 1e-9; y += spacing) {
    for (let x = 0; x <= width + 1e-9; x += spacing) {
      points.push({ x: Math.min(x, width), y: Math.min(y, height) });
    }
  }

  if (points[points.length - 1]?.x !== width || points[points.length - 1]?.y !== height) {
    points.push({ x: width, y: height });
  }

  return points;
}

function buildCandidates(width, height, radius) {
  const step = Math.max(radius * Math.sqrt(2), radius / 2);
  const candidates = [];
  const rows = Math.max(1, Math.ceil(height / step));
  const cols = Math.max(1, Math.ceil(width / step));

  for (let row = 0; row <= rows; row += 1) {
    const y = Math.min(row * step, height);
    const offset = row % 2 === 0 ? 0 : step / 2;
    for (let col = 0; col <= cols; col += 1) {
      const x = Math.min(col * step + offset, width);
      candidates.push({ x, y });
    }
  }

  const edgeStep = Math.max(radius, step);
  for (let x = 0; x <= width + 1e-9; x += edgeStep) {
    candidates.push({ x: Math.min(x, width), y: 0 });
    candidates.push({ x: Math.min(x, width), y: height });
  }
  for (let y = 0; y <= height + 1e-9; y += edgeStep) {
    candidates.push({ x: 0, y: Math.min(y, height) });
    candidates.push({ x: width, y: Math.min(y, height) });
  }

  candidates.push(
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: 0, y: height },
    { x: width, y: height },
    { x: width / 2, y: height / 2 },
  );

  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${round(candidate.x, 2)},${round(candidate.y, 2)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function coveredPointIndexes(candidate, points, radius) {
  const radiusSquared = radius * radius;
  const indexes = [];
  points.forEach((point, index) => {
    const dx = point.x - candidate.x;
    const dy = point.y - candidate.y;
    if (dx * dx + dy * dy <= radiusSquared + 1e-9) {
      indexes.push(index);
    }
  });
  return indexes;
}

function optimizeSensorPlacement(input = {}) {
  const width = parsePositiveNumber(input.width, 20);
  const height = parsePositiveNumber(input.height, 30);
  const radius = parsePositiveNumber(input.radius, 5);
  const resolution = parsePositiveNumber(input.resolution, Math.max(radius / 2, 1));
  const maxSensors = Math.max(1, Math.floor(parsePositiveNumber(input.maxSensors, 500)));

  const sampleSpacing = Math.min(resolution, radius);
  const points = buildGrid(width, height, sampleSpacing);
  const candidates = buildCandidates(width, height, radius);
  const coverageSets = candidates.map((candidate) => ({
    ...candidate,
    covers: coveredPointIndexes(candidate, points, radius),
  }));

  const uncovered = new Set(points.map((_, index) => index));
  const sensors = [];

  while (uncovered.size > 0 && sensors.length < maxSensors) {
    let best = null;
    let bestGain = 0;

    coverageSets.forEach((candidate) => {
      if (candidate.selected) return;
      let gain = 0;
      candidate.covers.forEach((index) => {
        if (uncovered.has(index)) gain += 1;
      });
      if (gain > bestGain) {
        best = candidate;
        bestGain = gain;
      }
    });

    if (!best || bestGain === 0) break;
    best.selected = true;
    best.covers.forEach((index) => uncovered.delete(index));
    sensors.push({
      id: sensors.length + 1,
      x: round(best.x),
      y: round(best.y),
      coveredSamples: bestGain,
    });
  }

  const coveredSamples = points.length - uncovered.size;
  const coverageRatio = points.length > 0 ? coveredSamples / points.length : 0;
  const theoreticalLowerBound = Math.ceil((width * height) / (Math.PI * radius * radius));

  return {
    input: {
      width,
      height,
      radius,
      resolution: sampleSpacing,
      maxSensors,
    },
    sensors,
    metrics: {
      sensorCount: sensors.length,
      sampleCount: points.length,
      coveredSamples,
      uncoveredSamples: uncovered.size,
      coverageRatio: round(coverageRatio, 4),
      theoreticalLowerBound,
      area: round(width * height),
      coverageAreaPerSensor: round(Math.PI * radius * radius),
    },
    uncoveredPoints: Array.from(uncovered).map((index) => ({
      x: round(points[index].x),
      y: round(points[index].y),
    })),
  };
}

module.exports = {
  optimizeSensorPlacement,
};
