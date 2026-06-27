import psycopg2
import uuid

class DB:

    def __init__(self):

        self.conn = psycopg2.connect(
            dbname="gemma",
            user="gemma",
            password="gemma",
            host="localhost"
        )

        self.cur = self.conn.cursor()

    # -------------------------
    # INSERT EXPERIMENT
    # -------------------------

    def insert_experiment(self, prompt, provider, max_tokens):

        eid = str(uuid.uuid4())

        self.cur.execute("""
            INSERT INTO experiments (id, prompt, provider, max_tokens)
            VALUES (%s, %s, %s, %s)
        """, (eid, prompt, provider, max_tokens))

        self.conn.commit()

        return eid

    # -------------------------
    # INSERT RUN
    # -------------------------

    def insert_run(self, experiment_id, provider, latency, tokens, cost, success):

        self.cur.execute("""
            INSERT INTO runs (experiment_id, provider, latency, tokens, cost, success)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (experiment_id, provider, latency, tokens, cost, success))

        self.conn.commit()
