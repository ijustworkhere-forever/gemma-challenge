CREATE TABLE experiments (
    id UUID PRIMARY KEY,
    prompt TEXT,
    provider TEXT,
    max_tokens INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE runs (
    id SERIAL PRIMARY KEY,
    experiment_id UUID,
    provider TEXT,
    latency FLOAT,
    tokens INT,
    cost FLOAT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    queue_depth INT,
    avg_latency FLOAT,
    total_cost FLOAT
);
