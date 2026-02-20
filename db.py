from cassandra.cluster import Cluster
import os

def get_session():
    # Uses the environment variables set in your CircleCI config
    contact_point = os.getenv("CASSANDRA_HOST", "localhost")
    port = int(os.getenv("CASSANDRA_PORT", 9042))
    
    cluster = Cluster([contact_point], port=port)
    session = cluster.connect()
    
    # Create keyspace if it doesn't exist (Cassandra requirement)
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS customer_segmentation 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
    """)
    session.set_keyspace('customer_segmentation')
    return session