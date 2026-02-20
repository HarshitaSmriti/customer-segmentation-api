import uuid
import os
from datetime import datetime
from cassandra.cluster import Cluster

def get_session():
    host = os.getenv("CASSANDRA_HOST", "localhost")
    port = int(os.getenv("CASSANDRA_PORT", 9042))
    
    cluster = Cluster([host], port=port)
    session = cluster.connect()
    
    # Setup Keyspace
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS customer_segmentation 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
    """)
    session.set_keyspace('customer_segmentation')
    
    # Setup Table
    session.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id uuid PRIMARY KEY,
            income float,
            age int,
            kmeans_cluster int,
            predicted_cluster int,
            created_at timestamp
        )
    """)
    return session

def save_prediction(data, k_cluster, p_cluster):
    try:
        session = get_session()
        query = """
            INSERT INTO predictions (prediction_id, income, age, kmeans_cluster, predicted_cluster, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        session.execute(query, (
            uuid.uuid4(),
            float(data.get('Income', 0)),
            int(data.get('Age', 0)),
            k_cluster,
            p_cluster,
            datetime.now()
        ))
    except Exception as e:
        print(f"Cassandra Save Error: {e}")