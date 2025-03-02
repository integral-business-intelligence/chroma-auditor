# Chroma Auditor

![Power Interface](/images/screenshot01.png)

## Overview

Chroma Auditor is a self-contained tool that enables business users to:
- Inspect and understand what's in their [Chroma](https://github.com/chroma-core/chroma) vector database
- See how their text is being chunked
- Organize chunks into logical groups (files and File Sets)
- Build confidence in RAG by providing traceability
- Have basic tools to fix/organize chunks if needed

Our philosophy here is to make this a relatively simple tool for a relatively simple user who may have limited programming experience.

### Target Audience

This tool is designed for users in light-commercial businesses that want visibility into how their documents are being processed in RAG systems. While these organizations may have technical support for setting up AI chat systems, end users appreciate transparency into document processing.

Chroma Auditor serves as a companion tool that lets business users inspect their Chroma database, understand document chunking and retrieval, and build confidence in AI chat response accuracy.

### Key Features

✅ GUI
- View all chunks (aka documents/entries/records/shards)
- Browse by file or File Set
- Export chunks to CSV
- Basic and Power user interfaces

✅ Metadata Management
- Add/delete metadata at chunk level
- Organize chunks into logical groups
- Track source files and File Sets
- Monitor embedding models used

✅ Database Operations
- Delete individual chunks
- Associate chunks with source files
- Group files into File Sets

✅ AI support features
- Document upload/ingestion via [Langflow](https://github.com/langflow-ai/langflow)
- Chat interface for RAG testing

## Screenshots

Power Interface:
![Power Interface](/images/screenshot01.png)

Basic Interface:
![Basic Interface](/images/screenshot02.png)

Upload Document:
![Upload Documents](/images/screenshot03.png)

Chat with Files:
![Chat Interface](/images/screenshot04.png)

## How it works

- The script is to be run server-side from the location of the Chroma database
- Gradio launches its own server
- Users can access the Gradio interface through a web browser by connecting to the host machine's IP and port
- The interface allows users to input a path to ChromaDB's persistent storage (db_path component)
- This path must be valid on the server machine where the script is running, not on the client's machine
- When users input a path and click "Load Database", the application creates a ChromaDB client
- All subsequent database operations use this client to interact with the ChromaDB instance
- Optionally: The script integrates with a running Langflow instance for document upload/embedding and chat functionality. Langflow must be running and accessible at the configured URL (default: http://127.0.0.1:7860, configurable)

## System Requirements

- Python 3.x
- Chroma
    - Setup for persistent storage / known file path
    - An existing collection
    - Custom component needed if you want to tag chunks with the name of the embedding model
    - Works with Chroma 0.6.3
- Gradio
	- Works with Gradio 5.15.0
- Langflow (optional but recommended)
	- Works with Langflow v1.1.4.post1
- Ollama (optional)
    - Embedding model
    - LLM


## Installation

### Simple Deployment / Quick-Start
If you just want to use the tool:

1. Download `chroma-auditor.py` onto the machine where your Chroma database is located
2. Make sure prerequisites are installed:
```bash
pip install chromadb gradio
```
1. Run the script:
```bash
python3 chroma-auditor.py
```

### Developer Setup
If you want to contribute or modify the codebase:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chroma-auditor.git
cd chroma-auditor
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

1. Configure the application:
- Edit CONFIGURATION section in chroma-auditor.py
- Set Chroma persistent storage path
- Configure Langflow connection (optional)

1. Start the application:
```bash
python3 chroma-auditor.py
```

### Langflow Integration Setup (optional)
If you want to use document upload and chat features:

1. Install and start Langflow:
```bash
uv pip install langflow
uv run langflow run
```

1. Configure Langflow pipelines:
   - In your Chroma components:
     - Set the persistent storage file path
     - Specify a collection name (e.g., 'langflow', 'main')
   - Create separate flows for document ingestion and chat
   - Test the flows independently in Langflow UI

2. Update Chroma Auditor configuration:
   - Set your Chroma persistent storage path
   - Configure Langflow connection:
     - API endpoint (default: http://127.0.0.1:7860)
     - File input endpoint
     - Chat input endpoint
     - Chat output endpoint

3. Start using the integrated features:
   - Run the script: `python3 chroma-auditor.py`
   - Navigate to the provided URL
   - Test document upload and chat functionality

Note: A collection must exist in Chroma before uploads can be processed through the Chroma Auditor interface.

## Configuration

In the script:
- Populate the CONFIGURATION section at the top of the script
- To configure Langflow (optional)
	- Populate the LANGFLOW CONFIGURATION section at the top of the script. This can be omitted if you do not intend to upload or query documents through Chroma Auditor.
	- A collection needs to exist in Chroma before uploads can be allowed via Chroma Auditor.

## Usage Guide

### Basic Interface Tab
- Connect to your Chroma database
- Browse all chunks in collection
- Add/delete metadata, export chunks, delete chunks

### Power Interface Tab
- Connect to your Chroma database
- Browse by file or File Set
- Add/delete metadata, export chunks, delete chunks

### Upload Documents Tab
1. Select source file for upload
2. Optionally assign to a File Set
3. Process through Langflow pipeline
4. Verify chunk creation and metadata

### Chat with Files Tab

- Select a specific file or File Set to chat with
- Enter your question in the chat input
- Test different prompts, observe responses, and identify potential issues with chunk size, content, or pipeline functionality

## Architecture

![Flow Diagram](/images/mermaid.png)

Chroma Auditor is designed as a monolithic application for simplicity and ease of deployment:

- Server-side script runs where Chroma database is located
- Gradio provides web interface accessible via browser
- Optional Langflow integration for document processing
- Single collection model ensures consistent RAG functionality
- Metadata-based organization rather than multiple collections

## Design Decisions and Trade-offs

### Monolithic Architecture

We intentionally designed Chroma Auditor as a single-file application, breaking from traditional multi-file Python architectures. This decision prioritizes:
- Simplicity of deployment (just copy one file)
- Ease of modification for non-technical users
- Reduced administrative overhead
- Straightforward troubleshooting
- AI-friendly codebase for future modifications

This approach means users only need to manage one file. This design choice prioritizes ease of deployment, modification, and troubleshooting over traditional software engineering principles. 

Users can download, run, and even make basic modifications to the script without needing to understand Python package structures or complex file dependencies. The script includes clear section headers and inline documentation to maintain readability despite its length.

The script was written in part by AI and the monolithic structure is also implemented to facilitate future modifications by AI - where the script can be loaded directly into an AI model for interpretation and modification.
### Single Collection Model
We deliberately limited the tool's document upload and RAG chat features to work only with a single Chroma collection. Viewing, editing metadata, deleting chunks, and exporting chunks is enabled for all collection in a Chroma instance. This decision was driven by:
- Chroma's limitation of not supporting cross-collection queries
- Preventing users from accidentally fragmenting their data
- Perserving RAG functionality across all documents
- Simplifying the mental model for business users

Instead of using multiple collections, we implemented File Sets through metadata, providing logical grouping while maintaining queryability across all documents.

### Langflow Integration
Rather than implementing ingestion directly, we delegate it to Langflow. This separation:
- Protects end users from accidentally modifying critical configurations (eg: changing embedding models)
- Ensures consistency in document processing
- Allows technical teams to modify pipelines without touching the interface
- Maintains a clear boundary between user operations and system configuration

Also:
- We setup this script to use one URL for an ingestion flow and a separate URL for retrieval flow. We found this improved stability and reliability.
### Interface Design Philosophy
The tool provides two interfaces that serve different purposes:

- Basic Interface: Mirrors Chroma's native hierarchy and terminology, allowing users to understand their data in terms of Chroma's documentation and mental model. While functional, this view is intentionally limited and primarily serves as an educational bridge to help users understand how Chroma organizes data.

- Power Interface: The primary workspace where users will spend most of their time. It provides intuitive, granular controls for managing chunks through file and File Set organizations, offering a more natural way to work with document-based data that aligns with how users think about their content.
	- A "file" in this interface is a filtering mechanism that filters the Chroma database to return only chunks with metadata key `source_file` and metadata value matching the value in the file dropdown. For example, the "Select File" dropdown shows the values associated with the `source_file` key in the database like, say, book.txt. When you select book.txt in the dropdown and load the file, the data frame will return all chunks tagged with "`source_file`:`book.txt`".
	- A "File Set" in this interface is a filtering mechanism that filters the Chroma database to return only chunks with metadata key `fileset` and metadata value matching the value in the file dropdown. For example, the "Select File Set" dropdown shows the values associated with the `fileset` key in the database like, say, `reading`. When you select `reading` in the dropdown and load the File Set, the data frame will return all chunks tagged with "`fileset`:`reading`". 
	- Individual chunks can be logically grouped as files and file sets using metadata operations

### Metadata Operations
- Add/delete
- Editing metadata is accomplished by deleting old and re-adding new metadata as-desired
- Add the metadata key "source_file" and a file name to append a source filename to your chunks. This will enable your files to be selectable in the file dropdown
- Add the metadata key "fileset" and a name for your File Set to group chunks into File Sets selectable in the dropdown
- Files can be removed from File Sets, and File Sets can be removed from the database, using the add and delete metadata functions in the interface

### Delete Chunks
- Deletes selected chunks from the database. Be careful there are no safeties.
- Deleting all chunks in a collection (emptying the collection) results in the collection being deleted, the HNSW segment directory (associated with the collection) being deleted to prevent it from becoming an orphan and causing conflicts, and the creation of a new collection with the same name (re-creation). So, the collection persists in name but, when documents are uploaded, a new HNSW segment directory will be created and become associated with this collection name.

### Export to Excel/CSV

Why Excel export:
- Quality Control & Validation:
	- Human reviewers can examine chunks in Excel/spreadsheet software and mark problematic chunks
	- Teams can collaborate on reviewing and annotating chunks
	- Easier to spot issues with chunk size, overlap, or content quality
- Cross-Platform Migration:
	- Easy way to move validated chunks between different vector DB implementations
	- Can serve as a backup format that's human-readable
	- Useful for transferring specific subsets of chunks between systems
- Metadata Management:
	- Bulk edit metadata in Excel then reimport
	- Add new metadata fields across many chunks
	- Clean up or standardize metadata across chunks
- Content Analysis:
	- Analyze chunk length distribution
	- Find duplicate or near-duplicate content
	- Review chunk boundaries and splitting quality
- Training Data Preparation:
	- Select high-quality chunks for fine-tuning models
	- Create gold-standard datasets for testing
	- Filter and clean training data
- Compliance & Auditing:
	- Document what content is in the vector database
	- Track changes and versions of chunks
### Document Uploading (bonus feature)
- Interface users can send files into an ingestion pipeline in Langflow where the result is vector embeddings stored in Chroma. Uploading via Chroma Auditor will automatically add metadata identifying the source file and, optionally, users can specify a File Set. 
- Text splitting technique and embedding model are managed in Langflow. This is discussed in our design philosophy section.
- Users can then confirm file integrity because every chunk has metadata showing the original file name, how many chunks were created, and the number of each chunk relative to the total (EG: 1 of 50, 2 of 50, 3 of 50 ... 50 of 50, etc.).
	- They can inspect every chunk and its associated metadata. 
	- They can add and delete individual key-value pairs of metadata on individual chunks and/or groups of chunks.
	- They can use metadata to add and remove individual files from File Sets (groups of files) and/or entirely remove files from their database by deleting all chunks derived from the file.
	- Metadata includes the embedding model that was used during ingestion

### Chat Interface (bonus feature)
- Can connect to Langflow to run RAG retrieval/chats. 
- Doing so applies metadata filtering in Chroma to enable users to chat with a specific file or a File Set.
- The chat interface is somewhat of a novelty/utility feature for users who may want to validate that their chunks are being returned in a retrieval pipeline. 
- For more advanced workflows, Chroma Auditor would just be used to assist in managing the database and users could ignore the chat tab and use other front ends, apps, workflows, etc. on top of Chroma - independent of the flows in Langflow that are setup to make the chat interface operational.

## Known Issues

1. Collection Corruption
	- Emptying collections can cause corruption of the Chroma collection
	- Symptom: the RAG chat returns errors indicating that the retrieved chunks are unreadable.
	- Workaround: delete the collection, create a new collection, and restart there. In other words, for now, don't empty out a collection unless you're willing to do this. Also, remember, Chroma Auditor works with one collection name specified in the Chroma component(s) in Langflow. So, if you do need to delete a collection and create a new one, you need to update these entries in Langflow if you want to upload documents and chat using the Chroma Auditor interface.

2. Database file path
	- The current code allows users to input any path on the server's file system. For a more secure deployment, you might want to implement authentication to control who can access the interface.

3. Metadata Handling
	- Tagging chunks with the embedding model name requires use of a custom Chroma component in Langflow.
	- All other metadata operations are handled with Chroma Auditor.
	- Deleting metadata is performed in two ways. When there is only a single key-value to delete from a chunk, we have only been able to delete via SQLite directly. When a key holds multiple values, we have been able to delete with the Chroma API.

## Future Development

- Collection statistics and metrics
- Source file archiving and retrieval
- User authentication system
- Multi-collection support
- Enhanced metadata management
- Bulk operations interface

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License

## Contact

dev@integralbi.ai

## Acknowledgments

- ChromaDB team
- Gradio team
- Langflow team
