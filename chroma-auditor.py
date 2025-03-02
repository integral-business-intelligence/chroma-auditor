############################## IMPORTS ##############################
import gradio as gr
import chromadb
import requests
import json
import os
import sys
import pandas as pd
import traceback
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
from gradio import State
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tempfile
import uuid

############################## CONFIGURATION ##############################
# Chroma Auditor Configuration
# --------------------------
# This section contains all configurable parameters for the application. Edit these
# values to match your environment before running the script.
#
# Optional: Set a default database path. If not set, users will need to input the path
# through the interface. This is useful for development or if you're deploying to a
# known environment.
DEFAULT_CHROMA_PATH = "/home/name/venvs/chroma/chroma_persistent_storage"


# Gradio Server Configuration (Gradio chooses automatically but can, alternatively, be specified here)
# ---------------------
#SERVER_HOST = "127.0.0.1"  # Where the Gradio interface will run
#SERVER_PORT = 7864        # Port for the Gradio interface

############################## LANGFLOW CONFIGURATION ##############################
# This Gradio interface is designed to work with Langflow, where the actual text splitting,
# embedding, and vector storage operations take place. Instead of implementing these operations
# directly in this script, we delegate them to a Langflow "flow" that should contain:
#
#   1. Text splitting configuration
#   2. Embedding model setup (e.g., via Ollama)
#   3. ChromaDB configuration (input database path and collection name)
# 
# The script serves as a user-friendly interface to these Langflow operations, making it
# reusable across different projects regardless of the specific embedding model or text
# processing parameters chosen in Langflow. This separation of concerns allows the interface
# to remain consistent while the underlying processing can be modified entirely within
# Langflow without requiring changes to this script.

LANGFLOW_API_URL = "http://127.0.0.1:7860"  # URL where your Langflow instance is running  
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "langflow")    # Default collection name

# Langflow Flow IDs
# ----------------
# You must replace these IDs with the ones from your Langflow instance.
# To find these IDs:
#   1. Open your Langflow UI
#   2. Navigate to your flows
#   3. The ID is in the URL when you open a flow
#   Example URL: http://localhost:7860/flow/ef65dcbe-6835-40ad-8ffc-958827d1cdbd
#                                          └────────── This is your Flow ID ──────┘

INGESTION_FLOW_ID = "ef65dcbe-6835-40ad-8ffc-958827d1cdbd"  # Flow for document ingestion
CHAT_FLOW_ID = "f9854cc7-15cb-469e-8d8a-4c44ebb287dc"      # Flow for chat/retrieval
INGESTION_FLOW_COLLECTION = "something"  # The name you hard-coded in the Langflow Chroma component

# Langflow Component IDs
# ---------------------
# These IDs correspond to specific components in your Langflow flows.
# You'll need to replace these with the IDs from your own Langflow setup.
# To find these IDs:
#   1. Open your Langflow flow
#   2. Click on the relevant component
#   3. Find the component ID in its properties panel

# Chat/Retrieval Flow Components

CHAT_INPUT_COMPONENT = "ChatInput-Dz1io"     # Chat input component sends user message into retrieval flow
CHROMA_QUERY_COMPONENT = "Chroma-JhFv9"      # Chroma query component in retrieval flow

# Document Upload/Ingestion Flow Components
FILE_INPUT_COMPONENT = "File-iN1hy"          # File input component sends file into ingestion flow
CHROMA_STORE_COMPONENT = "Chroma-XY1789"     # CHROMA_STORE_COMPONENT is not used in this code - files are sent to Langflow's ingestion flow through FILE_INPUT_COMPONENT, and Chroma storage is handled within the Langflow pipeline

############################## DATABASE LOADING FUNCTIONS ##############################
# Connects the interface to the database of interest so that its data can be accessed

def initialize_database(db_path=DEFAULT_CHROMA_PATH):
    """
    Initialize the database with the main collection if it doesn't exist.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection_name = os.getenv("DEFAULT_COLLECTION", INGESTION_FLOW_COLLECTION)
        client.get_or_create_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

def try_connect_db(path):
    """Attempt to connect to database with timeout"""
    return chromadb.PersistentClient(path=path)

def load_database(db_path=DEFAULT_CHROMA_PATH):
    """
    Initialize database connection with single toast notification.
    Updated for ChromaDB v0.6.0+ compatibility.
    """
    print(f"\nAttempting to load database from: {db_path}")
    
    try:
        # Check for empty path
        if not db_path:
            raise gr.Error("No database path provided")
            
        # Check if path exists
        if not os.path.exists(db_path):
            raise gr.Error(f"No database found at: {db_path}")

        # Check that it's a ChromaDB directory
        sqlite_path = os.path.join(db_path, "chroma.sqlite3")
        segments_path = os.path.join(db_path, ".chroma", "segments")
        
        if not (os.path.isfile(sqlite_path) or os.path.isdir(segments_path)):
            raise gr.Error(f"No valid ChromaDB database found at: {db_path}")
            
        # Try to connect with timeout
        try:
            print("Attempting to connect to database...")
            with ThreadPoolExecutor() as executor:
                future = executor.submit(chromadb.PersistentClient, path=db_path)
                client = future.result(timeout=2)
        except TimeoutError:
            raise gr.Error(f"Connection timeout for database at: {db_path}")
        except Exception as e:
            raise gr.Error(f"Invalid database at: {db_path}")

        # Get collections - Updated for v0.6.0
        collection_names = client.list_collections()
        print(f"Found collections: {collection_names}")
        
        # Handle empty database
        if not collection_names:
            return [
                gr.update(choices=[], value=None),  # collection_dropdown
                gr.update(visible=True),            # load_collection_btn
                gr.update(choices=[]),              # power_file_dropdown
                gr.update(choices=[]),              # power_fileset_dropdown
                gr.update(choices=[]),              # existing_fileset_dropdown
                gr.update(choices=[]),              # file_dropdown
                gr.update(choices=[]),              # fileset_dropdown
                DEFAULT_COLLECTION,                 # current_collection_display
                DEFAULT_COLLECTION,                 # current_collection_state
                "Warning: No collections found - database may be empty"  # status
            ]

        # Get initial collection name
        initial_collection = collection_names[0] if collection_names else DEFAULT_COLLECTION
        print(f"Selected initial collection: {initial_collection}")
            
        # Get files and filesets for the initial collection
        files = get_unique_filenames(db_path, initial_collection)
        filesets = get_filesets(db_path, initial_collection)
        
        print("Successfully loaded database")
        
        # Return successful initialization
        success_msg = f"✓ Successfully loaded database with {len(collection_names)} collections"
        return [
            gr.update(choices=collection_names, value=initial_collection),
            gr.update(visible=True),
            gr.update(choices=files),
            gr.update(choices=filesets),
            gr.update(choices=filesets),
            gr.update(choices=files),
            gr.update(choices=filesets),
            initial_collection,
            initial_collection,
            success_msg
        ]
        
    except gr.Error as e:
        # Let Gradio errors pass through
        raise e
    except Exception as e:
        # Wrap unexpected errors
        raise gr.Error(f"Error loading database: {str(e)}")

############################## FILESET MANAGEMENT FUNCTIONS ##############################
# Loads the list of File Sets which we define as groups of files/chunks bundled together with the
# metadata key "fileset". This enables a user to interact with multiples files, and their underlying
# chunks in an intuitive way (workings with files is intuitive for end users, working with chunks is not)

def get_filesets(db_path, collection_name):
    """Get unique filesets from a specific collection"""
    print(f"Looking for filesets in collection: {collection_name}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        
        filesets = set()
        for metadata in results['metadatas']:
            if metadata and 'fileset' in metadata:
                filesets.update(fs.strip() for fs in metadata['fileset'].split('|') if fs.strip())
        
        sets = sorted(list(filesets))
        print(f"Found filesets: {sets}")
        return sets
        
    except Exception as e:
        print(f"Error getting filesets: {str(e)}")
        return []

def load_fileset_documents(fileset_name: str, collection_name: str, db_path=DEFAULT_CHROMA_PATH) -> tuple:
    """
    Load fileset documents using client-side filtering with pipe-separated format.
    Updated for ChromaDB v0.6.0+ compatibility.
    """
    print(f"\nLOADING FILESET: {fileset_name}")
    print(f"Collection: {collection_name}")
    print(f"DB Path: {db_path}")
    
    try:
        if not fileset_name or not collection_name:
            return pd.DataFrame(), "Missing fileset name or collection name"

        client = chromadb.PersistentClient(path=db_path or DEFAULT_CHROMA_PATH)
        
        # Check if collection exists - v0.6.0 compatible
        collections = client.list_collections()
        if collection_name not in collections:
            print(f"Collection {collection_name} not found. Available collections: {collections}")
            return pd.DataFrame(), f"Collection '{collection_name}' does not exist"
            
        collection = client.get_collection(name=collection_name)
        
        # Get all documents and filter client-side for maximum compatibility
        results = collection.get()
        print(f"Retrieved {len(results['ids'])} total documents")
        
        matches = []
        for doc_id, document, metadata in zip(results['ids'], results['documents'], results['metadatas']):
            if metadata and 'fileset' in metadata:
                filesets = [fs.strip() for fs in metadata['fileset'].split('|')]
                if fileset_name in filesets:
                    matches.append((doc_id, document, metadata))

        if not matches:
            print(f"No matches found for fileset: {fileset_name}")
            return pd.DataFrame(), f"No documents found in fileset '{fileset_name}'"

        print(f"Found {len(matches)} matching documents")
        
        # Unpack matches and create DataFrame
        ids, docs, metadatas = zip(*matches)
        df = pd.DataFrame({
            'Selected': ['Not Selected'] * len(ids),
            'Metadata': [json.dumps(m, indent=2) for m in metadatas],
            'File Chunk': docs,
            'ID': ids
        })

        # Sort using metadata values
        try:
            df['source_file'] = df['Metadata'].apply(lambda x: json.loads(x).get('source_file', ''))
            df['chunk_index'] = df['Metadata'].apply(lambda x: json.loads(x).get('chunk_index', 0))
            df = df.sort_values(['source_file', 'chunk_index']).drop(columns=['source_file', 'chunk_index'])
            print(f"Successfully sorted {len(df)} documents")
        except Exception as sort_error:
            print(f"Warning: Could not sort documents: {sort_error}")
            # Continue without sorting if there's an error
        
        return df, f"Loaded {len(df)} chunks from '{fileset_name}'"
        
    except Exception as e:
        error_msg = f"Fileset load error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return pd.DataFrame(), error_msg

def refresh_all_filesets(db_path: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # Check if collection exists
        collections = client.list_collections()
        if collection_name not in [col.name for col in collections]:
            return [gr.update(choices=[])] * 3
            
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        
        filesets = set()
        for metadata in results['metadatas']:
            if metadata and 'fileset' in metadata:
                current_filesets = [fs.strip() for fs in metadata['fileset'].split('|') if fs.strip()]
                filesets.update(current_filesets)
        
        fileset_list = sorted(list(filesets))
        
        return [
            gr.update(choices=fileset_list, value=None),
            gr.update(choices=fileset_list, value=None),
            gr.update(choices=fileset_list, value=None)
        ]
    except Exception as e:
        print(f"Error refreshing filesets: {str(e)}")
        return [gr.update(choices=[])] * 3

############################## COLLECTION MANAGEMENT FUNCTIONS ##############################
# Pulls information about collections, which are groups of chunks as defined by Chroma, into 
# the UI for user interaction

def load_collection(collection_name: str, db_path=DEFAULT_CHROMA_PATH) -> pd.DataFrame:
    """Loads a collection and returns its contents as a DataFrame with complete metadata"""
    print(f"\n=== DEBUG: LOADING COLLECTION ===")
    print(f"Collection name: {collection_name}")
    print(f"DB Path: {db_path}")
    
    if not collection_name:
        print("No collection name provided")
        return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])
        
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=db_path)
        
        # Check if collection exists
        collection_names = client.list_collections()
        
        if collection_name not in collection_names:
            print(f"Collection {collection_name} not found. Available collections: {collection_names}")
            return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])
            
        # Get collection
        collection = client.get_collection(name=collection_name)
        
        # Get all items with all metadata
        print("\nFetching items from collection...")
        items = collection.get(
            include=['metadatas', 'documents', 'embeddings']
        )
        
        # Debug print the first item's metadata
        if items['metadatas'] and len(items['metadatas']) > 0:
            print("\nSample metadata from first item:")
            print(json.dumps(items['metadatas'][0], indent=2))
        
        if not items['ids']:
            print("No items found in collection")
            return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])
            
        # Create DataFrame
        print("\nCreating DataFrame...")
        data = {
            'Selected': ['Not Selected'] * len(items['ids']),
            'Metadata': [json.dumps(m, indent=2) if m else "{}" for m in items['metadatas']],
            'File Chunk': items['documents'],
            'ID': items['ids']
        }
        
        df = pd.DataFrame(data)
        
        # Debug print the first row's metadata after DataFrame creation
        if not df.empty:
            print("\nFirst row metadata in DataFrame:")
            print(df['Metadata'].iloc[0])
        
        # Sort if we have data
        if len(df) > 0:
            df['source_file'] = df['Metadata'].apply(lambda x: json.loads(x).get('source_file', '') if isinstance(x, str) else '')
            df['chunk_index'] = df['Metadata'].apply(lambda x: int(json.loads(x).get('chunk_index', 0)) if isinstance(x, str) else 0)
            df = df.sort_values(['source_file', 'chunk_index'])
            df = df.drop(['source_file', 'chunk_index'], axis=1)
        
        print(f"\nSuccessfully loaded {len(df)} chunks from collection")
        print("=== END DEBUG ===\n")
        
        return df.copy()
        
    except Exception as e:
        print(f"Error loading collection: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])

def update_collection_state(collection_name):
    """
    Update all components when collection selection changes
    """
    try:
        print(f"Updating collection state to: {collection_name}")
        
        # Update file and fileset lists for the new collection
        files = get_unique_filenames(db_path.value, collection_name)
        filesets = get_filesets(db_path.value, collection_name)
        
        return (
            collection_name,  # current_collection_state
            collection_name,  # current_collection_display
            gr.update(choices=files),  # power_file_dropdown
            gr.update(choices=filesets),  # power_fileset_dropdown
            gr.update(choices=filesets),  # existing_fileset_dropdown
            gr.update(choices=files),  # file_dropdown
            gr.update(choices=filesets),  # fileset_dropdown
        )
    except Exception as e:
        print(f"Error updating collection state: {str(e)}")
        return [gr.update()] * 7

############################## FILE MANAGEMENT FUNCTIONS ##############################
# Metadata is applied to chunks during processing to append the original filename.
# This allows end users to interact with representations of individual files without
# worrying directly about all of the underlying chunks.

def get_unique_filenames(db_path, collection_name):
    """Get unique filenames from a specific collection"""
    print(f"Looking for files in collection: {collection_name}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # Verify collection exists
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        
        unique_files = set()
        for metadata in results['metadatas']:
            if metadata and 'source_file' in metadata:
                unique_files.add(os.path.basename(metadata['source_file']))
        
        files = sorted(list(unique_files))
        print(f"Found files: {files}")
        return files
        
    except Exception as e:
        print(f"Error getting unique filenames: {str(e)}")
        return []

def load_file_chunks(db_path: str, filename: str, collection_name: str, key: str = "source_file") -> tuple:
    """
    Load chunks from a file with improved error handling and debugging.
    
    Args:
        db_path: Path to ChromaDB
        filename: Name of file to load
        collection_name: Name of collection to query
        key: Metadata key to search ("source_file" or "fileset")
    
    Returns:
        tuple: (DataFrame, status message)
    """
    print(f"\nLOADING CHUNKS:")
    print(f"DB Path: {db_path}")
    print(f"Filename: {filename}")
    print(f"Collection: {collection_name}")
    print(f"Key: {key}")

    if not filename:
        print("No filename provided")
        return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID']), "No file selected"

    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)  # Use the actual collection name
        
        # Construct where clause
        where_clause = {key: filename}
        print(f"Using where clause: {where_clause}")
        
        results = collection.get(where=where_clause)
        print(f"Found {len(results['ids'])} matching documents")

        if not results['ids']:
            print("No results found")
            return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID']), "No matching documents found"
        
        # Create DataFrame
        df = pd.DataFrame({
            'Selected': ['Not Selected'] * len(results['ids']),
            'Metadata': [json.dumps(m, indent=2) if m else "{}" for m in results['metadatas']],
            'File Chunk': results['documents'],
            'ID': results['ids']
        })

        # Sort based on the key type
        if key == "source_file":
            df['chunk_index'] = df['Metadata'].apply(
                lambda x: int(json.loads(x).get('chunk_index', 0))
            )
            df = df.sort_values('chunk_index', ascending=True)
            df = df.drop(columns=['chunk_index'])
        else:
            df['source_file'] = df['Metadata'].apply(
                lambda x: json.loads(x).get('source_file', '')
            )
            df['chunk_index'] = df['Metadata'].apply(
                lambda x: int(json.loads(x).get('chunk_index', 0))
            )
            df = df.sort_values(['source_file', 'chunk_index'])
            df = df.drop(columns=['source_file', 'chunk_index'])

        print(f"Successfully loaded {len(df)} chunks")
        return df.reset_index(drop=True), f"Successfully loaded {len(df)} chunks"
            
    except Exception as e:
        print(f"Error loading chunks: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID']), f"Error: {str(e)}"

def export_selected_chunks(selected_indices, df):
    """Create file for download"""
    try:
        if not selected_indices or df.empty:
            return None
            
        selected_rows = df.iloc[selected_indices]
        export_df = selected_rows.drop(columns=['Selected'])
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"selected_chunks_{timestamp}.csv"
        
        # Create full path in system's temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        
        # Save DataFrame to CSV
        export_df.to_csv(temp_path, index=False)
        
        return temp_path
        
    except Exception as e:
        print(f"Export error: {str(e)}")
        traceback.print_exc()
        return None

def handle_export_with_notification(indices, df):
    """Handle export and show user-facing notification"""
    if not indices:
        gr.Warning("No rows selected for export")
        return gr.update(value=None, visible=False)
        
    filepath = export_selected_chunks(indices, df)
    
    if filepath:
        filename = os.path.basename(filepath)
        gr.Success(f"📥 Download is Ready {filename}")
        return gr.update(value=filepath, visible=True)
    else:
        gr.Warning("Export failed")
        return gr.update(value=None, visible=False)

def handle_file_clear():
    """Hide file component when cleared"""
    return gr.update(value=None, visible=False)

############################## METADATA MANAGEMENT FUNCTIONS ##############################
# For adding and deleting metadata from individual chunks or groups of chunks. Fine-grained control.

def add_metadata(db_path, selected_indices, category, value, df, view_type, view_value, collection_name):
    """Adds metadata to selected chunks and reloads the correct view"""
    if not category or not value or not selected_indices:
        return df
    
    try:
        print(f"\nAdding metadata:")
        print(f"View Type: {view_type}")
        print(f"View Value: {view_value}")
        print(f"Collection: {collection_name}")
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        
        # Get IDs of selected rows
        selected_ids = [df.iloc[idx]['ID'] for idx in selected_indices]
        results = collection.get(ids=selected_ids)
        
        # Update metadata for each selected document
        for i, doc_id in enumerate(selected_ids):
            metadata = results['metadatas'][i] or {}
            
            # Special handling for filesets
            if category == "fileset":
                current_filesets = set()
                if "fileset" in metadata:
                    current_filesets.update(fs.strip() for fs in metadata["fileset"].split("|") if fs.strip())
                current_filesets.add(value)
                metadata["fileset"] = "|".join(sorted(current_filesets))
            else:
                metadata[category] = value
            
            collection.update(ids=[doc_id], metadatas=[metadata])
        
        print(f"Successfully updated metadata for {len(selected_ids)} documents")
        print(f"Reloading view type: {view_type}")
        
        # Reload based on view type
        if view_type == "file":
            updated_df, _ = load_file_chunks(
                db_path=db_path,
                filename=view_value,
                collection_name=collection_name,
                key="source_file"
            )
        elif view_type == "fileset":
            updated_df, _ = load_fileset_documents(
                fileset_name=view_value,
                collection_name=collection_name,
                db_path=db_path
            )
        else:  # collection view
            print("Loading entire collection view")
            updated_df = load_collection(collection_name, db_path)
            print(f"Loaded collection with {len(updated_df)} rows")
        
        # Maintain selection state
        if not updated_df.empty:
            updated_df['Selected'] = 'Not Selected'
            # Preserve selections if indices are valid
            valid_indices = [idx for idx in selected_indices if idx < len(updated_df)]
            if valid_indices:
                updated_df.iloc[valid_indices, updated_df.columns.get_loc('Selected')] = 'Selected'
        
        return updated_df
            
    except Exception as e:
        print(f"Add metadata error: {str(e)}")
        traceback.print_exc()
        return df

def delete_metadata(db_path, selected_indices, category, value, df, view_type, view_value, collection_name):
    """Deletes metadata from selected chunks and reloads the correct view"""
    print(f"\nDeleting metadata:")
    print(f"View Type: {view_type}")
    print(f"View Value: {view_value}")
    print(f"Collection: {collection_name}")
    
    if not category or not selected_indices or not collection_name:
        return df
        
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        
        # Get IDs of selected rows
        selected_ids = [df.iloc[idx]['ID'] for idx in selected_indices]
        results = collection.get(ids=selected_ids)

        # Set up SQLite connection
        import sqlite3
        conn = sqlite3.connect(f"{db_path}/chroma.sqlite3")
        cursor = conn.cursor()
            
        for i, doc_id in enumerate(selected_ids):
            metadata = results['metadatas'][i] or {}
            if category in metadata:
                current_value = metadata[category]
                
                # Handle pipe-delimited values with Chroma API
                if isinstance(current_value, str) and '|' in current_value:
                    values = [v.strip() for v in current_value.split('|')]
                    if value in values:
                        values.remove(value)
                        if values:
                            metadata[category] = '|'.join(values)
                            collection.update(ids=[doc_id], metadatas=[metadata])
                        else:
                            del metadata[category]
                            collection.update(ids=[doc_id], metadatas=[metadata])
                else:
                    # Single value - use SQLite
                    if str(current_value) == str(value):
                        cursor.execute(
                            "SELECT id FROM embeddings WHERE embedding_id = ?",
                            (doc_id,)
                        )
                        embedding_id = cursor.fetchone()[0]
                        
                        cursor.execute(
                            """
                            DELETE FROM embedding_metadata 
                            WHERE id = ? AND key = ? AND string_value = ?
                            """, 
                            (embedding_id, category, value)
                        )
                        print(f"SQL delete executed for {doc_id}")

        conn.commit()
        conn.close()

        # Reload based on view type
        if view_type == "file":
            updated_df, _ = load_file_chunks(
                db_path=db_path,
                filename=view_value,
                collection_name=collection_name,
                key="source_file"
            )
        elif view_type == "fileset":
            updated_df, _ = load_fileset_documents(
                fileset_name=view_value,
                collection_name=collection_name,
                db_path=db_path
            )
        else:  # collection view
            updated_df = load_collection(collection_name, db_path)
        
        # Maintain selection state
        if not updated_df.empty:
            updated_df['Selected'] = 'Not Selected'
            valid_indices = [idx for idx in selected_indices if idx < len(updated_df)]
            if valid_indices:
                updated_df.iloc[valid_indices, updated_df.columns.get_loc('Selected')] = 'Selected'
        
        return updated_df
            
    except Exception as e:
        print(f"Delete metadata error: {str(e)}")
        traceback.print_exc()
        return df

############################## SELECTION MANAGEMENT FUNCTIONS ##############################
# Sets up the ability for a user to click to select/deselect individual rows in a dataframe
# that present file chunks. This is how a user can act on individual chunks.

def update_selection_state(indices, df):
    """
    Updates the selection state in the dataframe with 'Selected' or 'Not Selected'
    instead of True/False
    
    Args:
        indices (list): List of selected row indices
        df (pd.DataFrame): The dataframe to update
        
    Returns:
        pd.DataFrame: Updated dataframe with new selection states
    """
    if df.empty:
        return df
    styled_df = df.copy()
    styled_df['Selected'] = ['Selected' if i in indices else 'Not Selected' 
                            for i in range(len(df))]
    return styled_df

def handle_select_all(df):
    """
    Selects all rows in the dataframe
    """
    indices = list(range(len(df)))
    return indices, update_selection_state(indices, df)

def handle_clear_selection(df):
    """
    Clears all selections in the dataframe
    """
    return [], update_selection_state([], df)

def handle_select(evt: gr.SelectData, state, df):
    """
    Handles individual row selection events
    """
    if evt.index[0] in state:
        state.remove(evt.index[0])
    else:
        state.append(evt.index[0])
    return state, update_selection_state(state, df)

def delete_entries(db_path, collection_name, selected_indices, df, view_type, view_value):
    """
    Deletes selected entries and reloads the appropriate view.
    For complete collection emptying, deletes and recreates the collection.
    """
    print(f"\nDeleting entries:")
    print(f"View Type: {view_type}")
    print(f"View Value: {view_value}")
    print(f"Collection: {collection_name}")
    
    if not selected_indices:
        return df, "No entries selected"
        
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(collection_name)
        
        # Get total count and selected IDs
        total_count = collection.count()
        selected_ids = [df.iloc[idx]['ID'] for idx in selected_indices if idx < len(df)]
        
        print(f"Total documents in collection: {total_count}")
        print(f"Number of documents to delete: {len(selected_ids)}")
        
        # Check if we're deleting all documents
        if len(selected_ids) >= total_count:
            print("DETECTED: Deleting all documents from collection!")
            
            try:
                # Reuse the exact same approach as your checker script
                # Get UUID before deleting collection
                import sqlite3
                conn = sqlite3.connect(f"{db_path}/chroma.sqlite3")
                cursor = conn.cursor()
                
                # Query for vector segment ID associated with this collection
                cursor.execute("""
                    SELECT s.id 
                    FROM segments s 
                    JOIN collections c ON s.collection = c.id 
                    WHERE c.name = ? AND s.scope = 'VECTOR'
                """, (collection_name,))
                
                uuid_dir = cursor.fetchone()
                conn.close()
                
                # Delete collection through API
                client.delete_collection(name=collection_name)
                print(f"Collection {collection_name} deleted")
                
                # Delete UUID directory if found
                if uuid_dir and uuid_dir[0]:
                    uuid_path = os.path.join(db_path, uuid_dir[0])
                    if os.path.exists(uuid_path):
                        import shutil
                        shutil.rmtree(uuid_path)
                        print(f"UUID directory {uuid_dir[0]} deleted")
                
                # Recreate the collection
                client.create_collection(name=collection_name)
                print(f"Collection {collection_name} recreated")
                
                status = "Collection has been completely reset (deleted and recreated)"
                
            except Exception as e:
                print(f"Error during collection reset: {e}")
                # Fallback to standard deletion if the reset fails
                collection.delete(ids=selected_ids)
                status = "Deleted documents using standard method"
        else:
            # Normal deletion for subset of documents
            collection.delete(ids=selected_ids)
            status = f"Deleted {len(selected_ids)} documents"
        
        # Reload based on view type
        if view_type == "file":
            updated_df, _ = load_file_chunks(
                db_path=db_path,
                filename=view_value,
                collection_name=collection_name,
                key="source_file"
            )
        elif view_type == "fileset":
            updated_df, _ = load_fileset_documents(
                fileset_name=view_value,
                collection_name=collection_name,
                db_path=db_path
            )
        else:  # collection view
            print("Loading entire collection view")
            updated_df = load_collection(collection_name, db_path)
            print(f"Loaded collection with {len(updated_df)} rows")
        
        return updated_df, status
        
    except Exception as e:
        print(f"Error deleting entries: {str(e)}")
        traceback.print_exc()
        return df, f"Error: {str(e)}"

############################## DOCUMENT UPLOAD FUNCTIONS ##############################
# Functions to allow users to submit a file into a text splitting/embedding/vector storage pipeline. 
# This section specifies how the file is transferred to the pipeline and how metadata is applied
# to the resulting chunks.

def process_file(file_obj, fileset_name):
    """
    Send file to Langflow's ingestion pipeline.
    Returns:
        tuple: (status message, file dropdown update, fileset dropdown update)
    """

    print("\n" + "="*50)
    print("START PROCESS FILE")
    print(f"File: {file_obj}")
    print(f"Fileset: {fileset_name}")
    print(f"Collection: {INGESTION_FLOW_COLLECTION}")
    print("="*50)

    try:
        if not file_obj:
            return "Error: No file uploaded"
            
        print(f"\nStarting file processing for {file_obj.name} in collection {INGESTION_FLOW_COLLECTION}")
        
        # Get initial count before processing
        client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
        collection = client.get_collection(name=INGESTION_FLOW_COLLECTION)
        
        print("\n=== CHECKING INITIAL STATE ===")
        initial_results = collection.get()
        initial_ids = set(initial_results['ids'])
        print(f"Initial collection count: {len(initial_ids)}")
        print(f"Initial IDs: {initial_ids}")
        
        # Read and encode the file
        with open(file_obj.name, "rb") as f:
            file_content = f.read()
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        
        # File data
        file_data = {
            "path": file_obj.name,
            "content": encoded_content,
            "name": os.path.basename(file_obj.name)
        }
        
        # Minimal payload - just the file data
        payload = {
            "input_value": json.dumps(file_data),
            "input_type": "text",
            "output_type": "text",
            "tweaks": {
                FILE_INPUT_COMPONENT: file_data,
            }
        }

        print("\n=== SENDING TO LANGFLOW ===")
        flow_response = requests.post(
            f"{LANGFLOW_API_URL}/api/v1/run/{INGESTION_FLOW_ID}?stream=false",
            json=payload
        )
        print(f"Langflow response status: {flow_response.status_code}")
        
        if flow_response.status_code == 200:
            print("\nFile sent successfully to Langflow")
            
            # Handle metadata after processing
            try:
                print("\n=== CHECKING FOR NEW CHUNKS ===")
                # Get results after processing
                time.sleep(2)  # Small delay to ensure processing is complete
                results = collection.get()
                current_ids = set(results['ids'])
                print(f"Current collection count: {len(current_ids)}")
                print(f"Current IDs: {current_ids}")
                
                # Find new document IDs
                new_ids = list(current_ids - initial_ids)
                print(f"\nNew chunks detected: {len(new_ids)}")
                print(f"New IDs: {new_ids}")
                
                if new_ids:
                    print("\n=== ADDING METADATA TO NEW CHUNKS ===")
                    # Get full data for new documents
                    new_results = collection.get(ids=new_ids)
                    total_new_chunks = len(new_ids)
                    print(f"Processing metadata for {total_new_chunks} chunks")
                    
                    # Update metadata only for new chunks
                    for i, doc_id in enumerate(new_ids):
                        metadata = new_results['metadatas'][i] if new_results['metadatas'][i] else {}
                        
                        # Base metadata that always gets added
                        base_metadata = {
                            'chunk_index': i + 1,
                            'total_chunks': total_new_chunks,
                            'source_file': os.path.basename(file_obj.name),
                            'upload_timestamp': datetime.now().isoformat()
                        }
                        
                        # Add fileset if provided
                        if fileset_name:
                            base_metadata['fileset'] = fileset_name
                        
                        metadata.update(base_metadata)
                        collection.update(ids=[doc_id], metadatas=[metadata])
                        print(f"Updated metadata for chunk {i+1}/{total_new_chunks}")
                    
                    # Return success message
                    return (f"File processed successfully into {total_new_chunks} chunks" + 
                           (f" and stored in fileset: {fileset_name}" if fileset_name else ""))
                else:
                    print("\nWARNING: No new chunks were detected after processing")
                    return "File processed but no new chunks were created"
                    
            except Exception as e:
                print(f"\nError adding metadata: {str(e)}")
                print(traceback.format_exc())
                return f"File processed but error adding chunk metadata: {str(e)}"
        else:
            print(f"\nLangflow error response: {flow_response.text}")
            return f"Error running flow: {flow_response.status_code} - {flow_response.text}"
    except Exception as e:
        print("\nException occurred:")
        print(traceback.format_exc())
        return f"Error processing file: {str(e)}"


def update_fileset_inputs(choice):
    if choice == "Create a new File Set":
        return gr.update(visible=True), gr.update(visible=False)
    elif choice == "Add to an existing File Set":
        filesets = get_filesets(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION)  # Use constant
        return gr.update(visible=False), gr.update(visible=True, choices=filesets)
    else:  # No File Set
        return gr.update(visible=False), gr.update(visible=False)
    
def handle_file_upload(file_obj, choice, new_name, existing_name):
    """
    Handle file upload to the main collection with optional fileset organization.
    """
    if not file_obj:
        return "Error: No file uploaded", gr.Dropdown()
    
    # If "No File Set" is selected, pass None as fileset_name
    if choice == "No File Set":
        fileset_name = None
    else:
        fileset_name = new_name if choice == "Create a new File Set" else existing_name
        if not fileset_name:
            return "Error: No file set name provided", gr.Dropdown()
        
    return process_file(file_obj, fileset_name)

############################## POWER INTERFACE SETUP ##############################

############################## CHAT INTERFACE SETUP ##############################
# These functions are for the chat tab in the Gradio interface. It allows user to 
# test whether their RAG pipeline is returning chat responses and whether the 
# responses reflect the chunks that they have identified. Users can identify specific
# files and filesets to be included in the RAG pipeline via filtering. 

def handle_chat_with_selection(message: str, history: list, selected_file: str, selected_fileset: str) -> tuple:
    """
    Enhanced chat handler with improved logging and request tracking
    """
    request_id = str(uuid.uuid4())[:8]  # Generate short request ID for tracking
    print(f"\n=== CHAT REQUEST [{request_id}] ===")
    print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"Message: {message}")
    print(f"Selected File: {selected_file}")
    print(f"Selected Fileset: {selected_fileset}")
    print(f"Using Collection: {INGESTION_FLOW_COLLECTION}")

    if not message.strip():
        print(f"[{request_id}] Empty message - skipping processing")
        return "", history, history, ""

    if not selected_file and not selected_fileset:
        print(f"[{request_id}] No file/fileset selected - returning early")
        return "", history, [*history, {"role": "assistant", "content": "Please select either a file or a file set to chat with."}], ""

    try:
        # Construct where clause for metadata filtering
        where_clause = None
        if selected_file:
            print(f"[{request_id}] Filtering by file: {selected_file}")
            where_clause = {"source_file": selected_file}
        elif selected_fileset:
            print(f"[{request_id}] Filtering by fileset: {selected_fileset}")
            where_clause = {"fileset": {"$contains": selected_fileset}}

        # Verify collection exists
        client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
        collections = client.list_collections()
        
        if INGESTION_FLOW_COLLECTION not in collections:
            error_msg = f"Error: Collection '{INGESTION_FLOW_COLLECTION}' not found"
            print(f"[{request_id}] {error_msg}")
            return "", history, [*history, {"role": "assistant", "content": error_msg}], error_msg

        # Construct payload
        payload = {
            "input_value": message,
            "input_type": "chat",
            "output_type": "chat",
            "tweaks": {
                CHAT_INPUT_COMPONENT: {
                    "input": message
                },
                CHROMA_QUERY_COMPONENT: {
                    "collection_name": INGESTION_FLOW_COLLECTION,
                    "persist_directory": DEFAULT_CHROMA_PATH,
                    "search_documents": {
                        "query": message,
                        "where": where_clause,
                        "n_results": 4,
                        "search_type": "similarity",
                        "include": ["documents", "metadatas", "distances"]
                    }
                }
            }
        }

        print(f"\n[{request_id}] Sending request to LangFlow")
        start_time = time.time()
        
        response = requests.post(
            f"{LANGFLOW_API_URL}/api/v1/run/{CHAT_FLOW_ID}?stream=false",
            json=payload,
            timeout=300
        ).json()

        print(f"[{request_id}] Response received in {(time.time() - start_time):.2f}s")
        
        # Extract response text and context
        response_text = None
        context_used = []
        
        if response and 'outputs' in response:
            outputs = response['outputs']
            if outputs and len(outputs) > 0:
                first_output = outputs[0].get('outputs', [{}])[0]
                
                if 'results' in first_output:
                    results = first_output['results']
                    if isinstance(results, dict):
                        message_obj = results.get('message', {})
                        if isinstance(message_obj, dict):
                            response_text = message_obj.get('text')
                        elif isinstance(message_obj, str):
                            response_text = message_obj
                        
                        context = results.get('context', [])
                        if context:
                            context_used = [
                                f"Source: {doc.get('metadata', {}).get('source_file', 'Unknown')}\n"
                                f"Chunk: {doc.get('page_content', '')[:200]}..."
                                for doc in context
                            ]

        if not response_text:
            print(f"[{request_id}] No valid response text found")
            response_text = "Error: No valid response received from the model"

        # Update history
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text}
        ]

        # Create context message
        context_msg = (
            f"Currently chatting with: "
            f"{'File: ' + selected_file if selected_file else 'File Set: ' + selected_fileset}\n"
        )
        print(f"[{request_id}] Request completed successfully")
        return "", updated_history, updated_history, context_msg

    except Exception as e:
        print(f"[{request_id}] Error in chat handler:")
        traceback.print_exc()
        error_msg = f"Error: {str(e)}"
        updated_history = [*history, {"role": "assistant", "content": error_msg}]
        return "", updated_history, updated_history, "Error occurred during chat"

############################## DEBUGGING FUNCTIONS ##########################

def show_toast(message):
    """Convert status messages to appropriate toast notifications"""
    print(f"Showing toast for message: {message}")  # Debug print
    
    if not message:
        return message
        
    if message.startswith("Error:"):
        gr.Warning(message)
    elif message.startswith("Warning:"):
        gr.Warning(message)
    elif message.startswith("Success:"):
        gr.Info(message)
    
    return message  # Return the message directly after showing notification

def check_collection_for_files(db_path: str, collection_name: str) -> tuple:
    """
    Check if a collection has any files (chunks with source_file metadata).
    Returns information about empty collections.
    
    Args:
        db_path: Path to ChromaDB
        collection_name: Name of collection to check
        
    Returns:
        tuple: (has_files: bool, message: str or None)
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        empty_collections = []
        
        # Check the specified collection
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        
        if not results['ids']:
            empty_collections.append(collection_name)
            return False, f"Warning: Collection '{collection_name}' is empty"
            
        # Check for source_file in metadata
        has_files = False
        for metadata in results['metadatas']:
            if metadata and 'source_file' in metadata:
                has_files = True
                break
                
        if not has_files:
            return False, f"Warning: No files found in collection '{collection_name}' - only chunks without source file metadata"
        
        return True, None  # No message needed when files are found
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def handle_collection_file_check(collection_name: str, db_path: str):
    """
    Secondary handler to check for files in collection and show warning if needed.
    Returns specific message about which collection is empty.
    """
    has_files, message = check_collection_for_files(db_path, collection_name)
    return message  # Will be None if collection contains files normally

############################## DROPDOWN REFRESH FUNCTIONS ##########################

def refresh_dropdowns():
    files = get_unique_filenames(db_path.value, current_collection_state.value)
    filesets = get_filesets(db_path.value, current_collection_state.value)
    return (gr.update(choices=files), gr.update(choices=filesets), 
            gr.update(choices=files), gr.update(choices=filesets))

############################## CSS STYLING ##############################
CUSTOM_CSS = """

/* Global background color */
body, .gradio-container, .contain {
    background-color: #fafafa !important;
}

/* Heading styles */
        .heading-2 {
            background-color: #ffffff !important;
            margin-bottom: -1px !important;  /* Remove default margins to work with card padding */
        }

/* ============================================================================
   CARD SYSTEM
   Define base card styles and size variations
   ============================================================================ */

/* Base card styles that all cards will share */
.card {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb;
    border-radius: 24px !important;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
}

/* Standard width card - locked at 700px */
.card-standard {
    width: 700px !important;
    max-width: 700px !important;
    min-width: 700px !important;
    margin-left: 0 !important;
}

/* Full width card - locked at 1425px */
.card-full {
    width: 1425px !important;
    max-width: 1425px !important;
    min-width: 1425px !important;
    margin-left: 0 !important;
}

/* Optional height constraints if needed */
.card-fixed-height {
    height: 200px !important;
    min-height: 200px !important;
    max-height: 200px !important;
}

/* Container for cards that sit side by side */
.card-container {
    display: flex !important;
    gap: 24px !important;
    justify-content: flex-start !important;
    align-items: stretch !important;
}

/* Class for formatted text inputs */
.formatted-textbox {
    margin-left: -11px !important;
    width: 101% !important;
    padding-right: 0px !important;
}

/* Class for formatting the middle metadata text boxes */
.formatted-middle-textbox {
    padding-left: 36px !important;
    padding-right: 8px !important;
}

/* Class for formatted buttons */
.formatted-button {
    border-radius: 8px !important;
    margin: 8px 0 !important;
    background-color: #E5E7EB !important;
    margin-right: 3px !important;
}

/* Class for formatted big buttons */
.formatted-button-big {
    max-width: 700px !important;
}

/* Row styling for buttons */
.button-row {
    justify-content: center !important;
    gap: 25px !important;
}

/* Style the input elements */
.scroll-hide, .wrap-inner {
    background-color: white !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
}

.card [style*="flex-grow"] {
    background-color: white !important;
}

/* These align the buttons for adding/removing metadata with the input boxes */
#metadata-row {
    display: flex;
    align-items: flex-end; /* Aligns all children (buttons) to the bottom */
}
.bottom-align {
    align-self: flex-end !important; /* Ensures specific buttons align to bottom */
    padding-bottom: 10px !important; /* Slightly lifts buttons */
    margin-bottom: 9px !important;
    margin-left: 16px !important;
    max-width: 96%; !important;
}

/* This removes some unwanted borders that appear on cards */
.card > div {
    background-color: white !important;
}

/* Set upload box container width */
.upload-box-size {
    max-width: 1425px; !important;
}

/* Set upload input textbox container widths */
.upload-input-size {
    max-width: 100%; !important;
}

/* Set upload button width */
.upload-button-size {
    max-width: 100%; !important;
    margin-right: 8px !important;
    margin-top: 32px !important;
    color: black !important;
}

.chat-button-align {
    align-self: flex-end !important; /* Ensures specific buttons align to bottom */
    padding-bottom: 10px !important; /* Slightly lifts buttons */
    margin-bottom: 9px !important;
    margin-left: 8px !important;
    max-width: 96%; !important;
}


.custom-sidebar {
    display: none !important;
}


/* Toast notification styling - fully opaque */
#toast-container > div[role="status"],
div[data-testid="notification"],
div[data-testid="notification-success"],
div[data-testid="notification-error"],
div[data-testid="notification-warning"],
div[data-testid="notification-info"] {
    background: white !important;
    background-color: white !important;
    opacity: 1 !important;
    -webkit-backdrop-filter: none !important;
    backdrop-filter: none !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    color: black !important;
}

/* Additional success style */
div[data-testid="notification-success"] {
    background: #4CAF50 !important;
    background-color: #4CAF50 !important;
    color: white !important;
}

/* Make sure the container itself is also opaque */
#toast-container {
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* Force any potential overlays to be opaque */
.toast-overlay,
.gradio-overlay {
    background: white !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    opacity: 1 !important;
}


/* Remove Gradio default footers */
        footer {
            display: none !important;
        }

"""

############################## GRADIO UI SETUP ##############################
# Displays four (4) tabs for user interaction. Power Interface. Basic Interface (reflects
# the basic controls via collections/chunks). Document Upload. Chat Interface.

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Chroma Auditor") as demo:

    current_view_type = gr.State("collection")
    current_view_value = gr.State(None)
    current_collection_state = gr.State(value=DEFAULT_COLLECTION)  # Moved here
    basic_selected_indices = gr.State([])
    power_selected_indices = gr.State([])
   
    with gr.Row():
        # Empty left column
        with gr.Column():
            pass
            
        # Right column with path and button
        with gr.Column(scale=1):
            
            with gr.Group(elem_classes="card card-standard"):

                gr.Markdown("## Database Connection", elem_classes="heading-2")

                db_path = gr.Textbox(
                    label="Input Path (EG: /home/name/chroma/persistent-storage)",
                    value=DEFAULT_CHROMA_PATH,
                    elem_classes="formatted-textbox"
                )

                with gr.Row():
                    load_db_btn = gr.Button("Load Database", elem_classes="formatted-button")

                collection_dropdown = gr.Dropdown(
                    label="Select Collection (after loading database)",
                    choices=[],
                    interactive=True,
                    visible=True,
                    allow_custom_value=True,
                    elem_classes="formatted-textbox"
                )

                current_collection_display = gr.Textbox(  # Add this component
                    label="Current Collection",
                    interactive=False,
                    visible=False  # Set to True if you want it visible by default
                )
            
                basic_selected_indices = gr.State([])  # Initialize empty state

    collection_status = gr.Textbox(label="Status", interactive=False, visible=False)
    
    with gr.Tabs() as tabs:
        # Power Interface Tab
        with gr.Tab("Power Interface", id=0):

            # State Management
            power_selected_indices = gr.State([])

            # Create a Row for the two main columns
            with gr.Row(equal_height=True):
                # Left Column - Files in Database
                with gr.Column(scale=1, min_width=700):
                    with gr.Group(elem_classes="card card-standard"):  # Added this wrapper
                        gr.Markdown("## Files in Database", elem_classes="heading-2")
                        power_file_dropdown = gr.Dropdown(label="Select File", choices=[], interactive=True, visible=True, elem_classes="formatted-textbox")
                        with gr.Row():
                            power_load_file_btn = gr.Button("Load File for Viewing", elem_classes="formatted-button")

                # Right Column - File Sets
                with gr.Column(scale=1, min_width=700):
                    with gr.Group(elem_classes="card card-standard"):
                        gr.Markdown("## File Sets in Database", elem_classes="heading-2")
                        power_fileset_dropdown = gr.Dropdown(label="Select File Set", choices=[], interactive=True, visible=True, elem_classes="formatted-textbox")
    
                        with gr.Row():
                            load_fileset_btn = gr.Button("Load File Set for Viewing", elem_classes="formatted-button")
            
                collection_dropdown.change(
                    fn=lambda collection_name: [
                        collection_name,  # current_collection_state
                        collection_name,  # current_collection_display
                        gr.update(choices=get_unique_filenames(DEFAULT_CHROMA_PATH, collection_name), value=None),  # power_file_dropdown
                        gr.update(choices=get_filesets(DEFAULT_CHROMA_PATH, collection_name), value=None)  # power_fileset_dropdown
                    ],
                    inputs=[collection_dropdown], 
                    outputs=[
                        current_collection_state,
                        current_collection_display,
                        power_file_dropdown,
                        power_fileset_dropdown
                    ]
                )

                collection_status.change(
                    fn=show_toast,
                    inputs=collection_status,
                    outputs=collection_status
                )

            gr.HTML('<div style="margin-top: 10px;"></div>')
            
            # Selection Buttons Row
            with gr.Group(elem_classes="card card-full"):
                gr.Markdown("## Details of Loaded Files and File Sets", elem_classes="heading-2")
                with gr.Row(elem_classes="button-row"):
                    power_select_all_btn = gr.Button("Select All", elem_classes="formatted-button formatted-button-big")
                    power_clear_selection_btn = gr.Button("Clear Selection", elem_classes="formatted-button formatted-button-big")
    
                # Dataframe Display
                power_chunk_display = gr.Dataframe(
                    headers=['Selected', 'Metadata', 'File Chunk', 'ID'],
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])
                )   

                power_chunk_display.select(
                    fn=handle_select,
                    inputs=[power_selected_indices, power_chunk_display],
                    outputs=[power_selected_indices, power_chunk_display]
                )

                power_chunks_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False
                )

                power_load_file_btn.click(
                    fn=lambda db, file, collection: (
                        *load_file_chunks(db_path=db, filename=file, collection_name=collection, key="source_file"),
                        [],
                        "file",    # view type
                        file       # view value
                    ),
                    inputs=[db_path, power_file_dropdown, current_collection_state],
                    outputs=[power_chunk_display, power_chunks_status, power_selected_indices, current_view_type, current_view_value]
                )

                load_fileset_btn.click(
                    fn=lambda db, fileset, collection: (
                        *load_fileset_documents(
                            fileset_name=fileset,
                            collection_name=collection,
                            db_path=db
                        ),
                        [],  # Reset selected indices
                        "fileset",  # view type
                        fileset    # view value
                    ),
                    inputs=[db_path, power_fileset_dropdown, current_collection_state],
                    outputs=[
                        power_chunk_display,  # DataFrame display
                        power_chunks_status,  # Status message
                        power_selected_indices,  # Reset selection
                        current_view_type,   # Update view type
                        current_view_value   # Update view value
                    ]
                )

                power_select_all_btn.click(
                    handle_select_all,
                    inputs=[power_chunk_display],
                    outputs=[power_selected_indices, power_chunk_display]
                )

                power_clear_selection_btn.click(
                    fn=handle_clear_selection,
                    inputs=[power_chunk_display],
                    outputs=[power_selected_indices, power_chunk_display]
                )

            # Metadata Management Section
            gr.HTML('<div style="margin-top: 10px;"></div>')
            with gr.Group(elem_classes="card card-full"):
                gr.Markdown("## Metadata Management", elem_classes="heading-2")
                with gr.Row(elem_id="metadata-row"):
                    with gr.Column(scale=1):
                        power_metadata_category = gr.Textbox(label="Metadata Category", elem_classes="formatted-textbox")
                    with gr.Column(scale=1):
                        power_metadata_value = gr.Textbox(label="Metadata Value", elem_classes="formatted-textbox formatted-middle-textbox")
                # Add Metadata Button 
                    with gr.Column(scale=1):
                        power_add_metadata_btn = gr.Button("Add Metadata to Selection", elem_classes="formatted-button bottom-align")

                        power_add_metadata_btn.click(
                            fn=lambda *args: [add_metadata(*args), *refresh_dropdowns()],
                            inputs=[
                                db_path,
                                power_selected_indices,
                                power_metadata_category,
                                power_metadata_value,
                                power_chunk_display,
                                current_view_type,
                                current_view_value,
                                current_collection_state
                            ],
                            outputs=[
                                power_chunk_display,
                                power_file_dropdown,
                                power_fileset_dropdown
                            ]
                        )

                with gr.Row(elem_id="metadata-row"):
                    with gr.Column(scale=1):
                        power_delete_metadata_category = gr.Textbox(label="Category to Delete", elem_classes="formatted-textbox")
                    with gr.Column(scale=1):
                        power_delete_metadata_value = gr.Textbox(label="Value to Delete", elem_classes="formatted-textbox formatted-middle-textbox")
                    # Delete Metadata Button
                    with gr.Column(scale=1):
                        power_delete_metadata_btn = gr.Button("Delete Metadata from Selection", elem_classes="formatted-button bottom-align")

                        power_delete_metadata_btn.click(
                            fn=lambda *args: [
                                delete_metadata(*args),  # First return value: updated dataframe
                                *refresh_dropdowns()     # Additional return values: updated dropdown choices
                            ],
                            inputs=[
                                db_path,
                                power_selected_indices,
                                power_delete_metadata_category,
                                power_delete_metadata_value,
                                power_chunk_display,
                                current_view_type,
                                current_view_value,
                                current_collection_state
                            ],
                            outputs=[
                                power_chunk_display,   # Updated dataframe
                                power_file_dropdown,   # Refreshed file list
                                power_fileset_dropdown # Refreshed fileset list
                            ]
                        )

            # Delete Entries Section
                with gr.Row():
                    power_delete_entries_btn = gr.Button("⚠️ Delete Selected File Chunk / Rows ⚠️", variant="stop", elem_classes="formatted-button")

                    power_delete_entries_btn.click(
                        fn=lambda *args: [delete_entries(*args)[0], *refresh_dropdowns()],  # Take just the DataFrame from the tuple
                        inputs=[
                            db_path,
                            current_collection_state,
                            power_selected_indices,
                            power_chunk_display,
                            current_view_type,
                            current_view_value
                        ],
                        outputs=[
                            power_chunk_display,
                            power_file_dropdown,
                            power_fileset_dropdown
                        ]
                    )

            # Export to Excel CSV section
                with gr.Row(elem_classes="button-row"):
                    power_export_btn = gr.Button("Export Selected Chunks to .csv (Excel format)", elem_classes="formatted-button")
                with gr.Row():
                    power_export_file = gr.File(
                        label="Download Exported Chunks",
                        visible=False,
                        interactive=True,
                        type="filepath"
                    )
                    power_export_file.clear(
                        fn=handle_file_clear,
                        outputs=power_export_file
                    )

                    power_export_btn.click(
                        fn=handle_export_with_notification,
                        inputs=[power_selected_indices, power_chunk_display],
                        outputs=power_export_file,
                        api_name="download_chunks"
                    )

            gr.HTML('<div style="margin-top: 30px;"></div>')

        # Basic Interface Tab
        with gr.Tab("Basic Interface"):         
            basic_selected_indices = gr.State([])

            with gr.Group(elem_classes="card card-full"):
                with gr.Row():
                    
                    #Button to load entire collection for viewing
                    load_collection_btn = gr.Button("View Entire Collection", visible=True, elem_classes="formatted-button")

            gr.HTML('<div style="margin-top: 30px;"></div>')

            with gr.Group(elem_classes="card card-full"):
                gr.Markdown("## Collection Details", elem_classes="heading-2")
                with gr.Row(elem_classes="button-row"):
                    basic_select_all_btn = gr.Button("Select All File Chunk / Rows", elem_classes="formatted-button formatted-big-button")
                    basic_clear_selection_btn = gr.Button("Clear Selections", elem_classes="formatted-button formatted-big-button")
    
                basic_collection_data = gr.Dataframe(
                    headers=['Selected', 'Metadata', 'File Chunk', 'ID'],
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=['Selected', 'Metadata', 'File Chunk', 'ID'])
                )

                load_collection_btn.click(
                    fn=lambda db_path, collection: (
                        load_collection(collection, db_path),  # Note the order of parameters
                        [],  # Reset selected indices
                        "collection",  # view type
                        collection,    # view value
                        f"Loaded collection: {collection}"  # status message
                    ),
                    inputs=[db_path, current_collection_state],
                    outputs=[
                        basic_collection_data,
                        basic_selected_indices,
                        current_view_type,
                        current_view_value,
                        collection_status
                    ]
                )

                basic_collection_data.select(
                    fn=handle_select,
                    inputs=[basic_selected_indices, basic_collection_data],
                    outputs=[basic_selected_indices, basic_collection_data]
                )

                basic_select_all_btn.click(
                    fn=handle_select_all,
                    inputs=[basic_collection_data],
                    outputs=[basic_selected_indices, basic_collection_data]
                )

                basic_clear_selection_btn.click(
                    fn=handle_clear_selection,
                    inputs=[basic_collection_data],
                    outputs=[basic_selected_indices, basic_collection_data]
                )

            # Metadata Management Section
            gr.HTML('<div style="margin-top: 30px;"></div>')
            with gr.Group(elem_classes="card card-full"):
                gr.Markdown("## Metadata Management", elem_classes="heading-2")

                with gr.Row(elem_id="metadata-row"):
                    basic_metadata_category = gr.Textbox(label="Metadata Category", elem_classes="formatted-textbox")
                    basic_metadata_value = gr.Textbox(label="Metadata Value", elem_classes="formatted-textbox formatted-middle-textbox")
                    basic_add_metadata_btn = gr.Button("Add Metadata to Selection", elem_classes="formatted-button bottom-align")
    
                with gr.Row(elem_id="metadata-row"):
                    basic_delete_metadata_category = gr.Textbox(label="Category to Delete", elem_classes="formatted-textbox")
                    basic_delete_metadata_value = gr.Textbox(label="Value to Delete", elem_classes="formatted-textbox formatted-middle-textbox")
                    basic_delete_metadata_btn = gr.Button("Delete Metadata from Selection", elem_classes="formatted-button bottom-align")

            # Delete Entries Section
                with gr.Row():
                    basic_delete_entries_btn = gr.Button("⚠️ Delete Selected File Chunk / Rows ⚠️", variant="stop", elem_classes="formatted-button")

            # Export to Excel CSV section
                with gr.Row(elem_classes="button-row"):
                    basic_export_btn = gr.Button("Export Selected Chunks to .csv (Excel format)", elem_classes="formatted-button")
                with gr.Row():
                    basic_export_file = gr.File(
                        label="Download Export",
                        visible=False,
                        interactive=True,
                        type="filepath"
                    )
                    basic_export_file.clear(
                        fn=handle_file_clear,
                        outputs=basic_export_file
                    )
                    basic_export_btn.click(
                        fn=handle_export_with_notification,
                        inputs=[basic_selected_indices, basic_collection_data],
                        outputs=basic_export_file
                    )

                # Metadata Management
                basic_add_metadata_btn.click(
                    fn=lambda *args: [add_metadata(*args), *refresh_dropdowns()],
                    inputs=[
                        db_path,
                        basic_selected_indices,
                        basic_metadata_category,
                        basic_metadata_value,
                        basic_collection_data,
                        current_view_type,
                        current_view_value,
                        current_collection_state
                    ],
                    outputs=[
                        basic_collection_data,
                        power_file_dropdown,
                        power_fileset_dropdown
                    ]
                )

                basic_delete_metadata_btn.click(
                    fn=lambda *args: [
                        delete_metadata(*args),  # First return value: updated dataframe
                        *refresh_dropdowns()     # Additional return values: updated dropdown choices
                    ],
                    inputs=[
                        db_path,                         # db_path
                        basic_selected_indices,          # selected_indices
                        basic_delete_metadata_category,  # category
                        basic_delete_metadata_value,     # value
                        basic_collection_data,           # df
                        current_view_type,              # view_type
                        current_view_value,             # view_value
                        current_collection_state        # collection_name
                    ],
                    outputs=[
                        basic_collection_data,    # Updated dataframe
                        power_file_dropdown,      # Refreshed file list
                        power_fileset_dropdown    # Refreshed fileset list
                    ]
                )

                basic_delete_entries_btn.click(
                    fn=lambda *args: [delete_entries(*args)[0], *refresh_dropdowns()],  # Take just the DataFrame from the tuple
                    inputs=[
                        db_path,
                        current_collection_state,
                        basic_selected_indices,
                        basic_collection_data,
                        current_view_type,
                        current_view_value
                    ],
                    outputs=[
                        basic_collection_data,
                        power_file_dropdown,
                        power_fileset_dropdown
                    ]
                )

            gr.HTML('<div style="margin-top: 30px;"></div>')

        # Upload Document Tab
        with gr.Tab("Upload Document"):

            
                # Center column container with 40% width
                with gr.Column(elem_classes="upload-box-size"):
                    # File upload at the top
                    file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".pdf", ".doc", ".docx"]
                    )

                with gr.Group(elem_classes="card card-full"):
                    gr.Markdown(f"## Add uploaded file to File Set", elem_classes="heading-2")
                    gr.Markdown(f"## Uploading is limited to only the collection: **{INGESTION_FLOW_COLLECTION}**", elem_classes="heading-2")

                    # File Set Options radio buttons
                    fileset_choice = gr.Radio(
                        choices=["Create a new File Set", "Add to an existing File Set", "No File Set"],
                        label="Options",
                        value="Create a new File Set",
                        elem_classes="formatted-textbox upload-input-size"
                    )
                
                    # New File Set Name textbox (visible by default)
                    new_fileset_name = gr.Textbox(
                        label="New File Set Name",
                        placeholder="Enter new file set name...",
                        visible=True,
                        interactive=True,
                        elem_classes="formatted-textbox upload-input-size"
                    )
                
                    # Existing File Set dropdown (hidden by default)
                    existing_fileset_dropdown = gr.Dropdown(label="Select Existing File Set", choices=[], interactive=True, visible=False)
                
                    # Process-the-uploaded-document button
                    with gr.Row():
                        upload_btn = gr.Button(
                            "Process and Store Document",
                            variant="primary",
                            elem_classes="formatted-button upload-button-size"
                        )

                    # Upload status textbox
                    upload_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="formatted-textbox upload-input-size"
                    )

                    # Triggers when the radio button selection changes between "Create a New File Set", "Add to Existing", etc.
                    fileset_choice.change(
                        fn=update_fileset_inputs,
                        inputs=[fileset_choice],
                        outputs=[new_fileset_name, existing_fileset_dropdown]
                    )

        # Chat with Files Tab

        with gr.Tab("Chat with Files"):
            with gr.Group(elem_classes="card card-full"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Select Content to Chat With", elem_classes="heading-2")
                    
                with gr.Row():
                    with gr.Column(scale=1):
                        file_dropdown = gr.Dropdown(label="Select Individual File", choices=[], interactive=True, visible=True, elem_classes="formatted-textbox")
                    with gr.Column(scale=1):
                        fileset_dropdown = gr.Dropdown(label="Select File Set", choices=[], interactive=True, visible=True, elem_classes="formatted-textbox")
            
                with gr.Row():
                    refresh_btn = gr.Button("↻ Refresh Lists", elem_classes="formatted-button")
                    clear_selection_btn = gr.Button("Clear Selection", elem_classes="formatted-button")
            
                    refresh_btn.click(
                        fn=lambda db, collection: [
                        gr.Dropdown(choices=get_unique_filenames(db, collection)),
                        gr.Dropdown(choices=get_filesets(db, collection))
                        ],
                        inputs=[db_path, current_collection_state],
                        outputs=[file_dropdown, fileset_dropdown]
                    )

                context_display = gr.Markdown("")
            
                chat_state = gr.State([])

                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    container=True,
                    bubble_full_width=False,
                    show_label=False,
                    layout="bubble",
                    type="messages"
                )
    
                with gr.Row():
                    with gr.Column(scale=8):
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send", elem_classes="formatted-button chat-button-align")
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Clear Chat", elem_classes="formatted-button chat-button-align")
            
                # Use a single event handler for both submit and click
                msg.submit(
                    fn=handle_chat_with_selection,
                    inputs=[msg, chat_state, file_dropdown, fileset_dropdown],
                    outputs=[msg, chat_state, chatbot, context_display]
                )

                send_btn.click(
                    fn=handle_chat_with_selection,  # Use same handler directly
                    inputs=[msg, chat_state, file_dropdown, fileset_dropdown],
                    outputs=[msg, chat_state, chatbot, context_display]
                )

                clear_btn.click(
                    fn=lambda: ([], [], [], ""),
                    outputs=[chat_state, chatbot, msg, context_display]
                )

                # Modified file/fileset selection handlers to prevent circular updates
                def update_file_dropdown(file_val, fileset_val):
                    """Prevent circular updates between dropdowns"""
                    if file_val and fileset_val:
                        return gr.update(value=None)
                    return gr.update()

                file_dropdown.change(
                    fn=update_file_dropdown,
                    inputs=[file_dropdown, fileset_dropdown],
                    outputs=[fileset_dropdown]
                )

                fileset_dropdown.change(
                    fn=update_file_dropdown,
                    inputs=[fileset_dropdown, file_dropdown],
                    outputs=[file_dropdown]
                )            

            gr.HTML('<div style="margin-top: 30px;"></div>')

            upload_btn.click(
                fn=lambda file_obj, choice, new_name, existing_name: [
                    status := process_file(file_obj, 
                                new_name if choice == "Create a new File Set" 
                                else existing_name if choice == "Add to an existing File Set" 
                                else None, 
                    ),
                    print(f"Process complete with status: {status}"),  # Debug print
                    status,  # Return status message
                    gr.update(choices=get_unique_filenames(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION)),
                    gr.update(choices=get_filesets(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION)),
                    gr.update(choices=get_filesets(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION)),
                    gr.update(choices=get_unique_filenames(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION)),
                    gr.update(choices=get_filesets(DEFAULT_CHROMA_PATH, INGESTION_FLOW_COLLECTION))
                ][-6:],  # Take only the last 6 items as our return values
                inputs=[
                    file_upload,
                    fileset_choice,
                    new_fileset_name,
                    existing_fileset_dropdown
                ],
                outputs=[
                    upload_status,
                    power_file_dropdown,
                    power_fileset_dropdown,
                    existing_fileset_dropdown,
                    file_dropdown,
                    fileset_dropdown
                ]   
            )    

################################ Event Handler Section #################################################

# Used at time of selecting a collection to determine whether it contains any chunks tagged as files
    collection_dropdown.change(
        fn=handle_collection_file_check,
        inputs=[collection_dropdown, db_path],
        outputs=collection_status
    )

# Database Loading Events
    load_db_btn.click(
        fn=load_database,
        inputs=[db_path],
        outputs=[
            collection_dropdown,
            load_collection_btn,
            power_file_dropdown,
            power_fileset_dropdown,
            existing_fileset_dropdown,
            file_dropdown,
            fileset_dropdown,
            current_collection_display,
            current_collection_state,
            collection_status
        ],
        queue=False,
        show_progress=False
    )

############################## MAIN ##############################

if __name__ == "__main__":
    initialize_database(DEFAULT_CHROMA_PATH)
    demo.launch(debug=True, show_error=True)
