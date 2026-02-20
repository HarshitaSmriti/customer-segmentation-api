import os
from cassandra.cluster import Cluster

def get_session():
    host = os.getenv("CASSANDRA_HOST", "127.0.0.1")
    port = int(os.getenv("CASSANDRA_PORT", 9042))

    cluster = Cluster([host], port=port)
    session = cluster.connect()
    return session