from cassandra.cluster import Cluster
import os

def get_session():
    # Get connection details from environment (set in CircleCI or .env)
    contact_point = os.getenv("CASSANDRA_HOST", "localhost")
    port = int(os.getenv("CASSANDRA_PORT", 9042))
    
    # Initialize the cluster and connect
    cluster = Cluster([contact_point], port=port)
    session = cluster.connect()
    
    # Assignment Requirement: Use Cassandra Keyspace
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS customer_segmentation 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
    """)
    session.set_keyspace('customer_segmentation')
    
    return session