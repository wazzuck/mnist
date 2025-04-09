-- Clear any existing database
DROP DATABASE IF EXISTS predictions;

-- Creates table on first run (idempotent)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    predicted INTEGER NOT NULL,
    true_label INTEGER
);

-- Optional: Add indexes or seed data
CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
