# Chroma Auditor Function Documentation

## Database Management Functions

### initialize_database
- **Status**: Active
- **Purpose**: Creates the initial ChromaDB structure if it doesn't exist
- **Behavior**: Creates a PersistentClient and ensures a default collection exists
- **Usage**: Called at application startup

### try_connect_db
- **Status**: Active
- **Purpose**: Helper function that attempts database connection
- **Behavior**: Creates a PersistentClient with timeout handling
- **Usage**: Used by load_database function

### load_database
- **Status**: Active
- **Purpose**: Main database loading function with comprehensive validation
- **Behavior**: 
  - Verifies database path and structure
  - Checks for required ChromaDB files
  - Establishes connection with timeout
  - Retrieves collections
  - Updates UI components
- **Usage**: Called when user clicks "Load Database" button

## Fileset Management Functions

### get_filesets
- **Status**: Active
- **Purpose**: Retrieves all unique fileset names from a collection
- **Behavior**: 
  - Examines all documents for 'fileset' metadata
  - Handles pipe-separated fileset values
  - Returns sorted list of unique names
- **Usage**: Used to populate fileset dropdown menus

### load_fileset_documents
- **Status**: Active
- **Purpose**: Loads all documents belonging to a specific fileset
- **Behavior**: 
  - Retrieves documents with matching fileset metadata
  - Creates DataFrame with document data
  - Sorts by source file and chunk index
- **Usage**: Core function for fileset operations

### refresh_all_filesets
- **Status**: Active
- **Purpose**: Updates UI components with current fileset data
- **Behavior**: Refreshes multiple dropdown menus with updated fileset list
- **Usage**: Called after fileset modifications

## Collection Management Functions

### load_collection
- **Status**: Active
- **Purpose**: Loads entire collection into DataFrame
- **Behavior**: 
  - Retrieves all documents, metadata, and embeddings
  - Creates structured DataFrame
  - Includes debug logging
- **Usage**: Used in basic interface for collection viewing

### update_collection_state
- **Status**: Active
- **Purpose**: Updates UI when collection selection changes
- **Behavior**: Refreshes file lists and fileset lists for new collection
- **Usage**: Collection dropdown change handler

## File Management Functions

### get_unique_filenames
- **Status**: Active
- **Purpose**: Gets list of unique source filenames
- **Behavior**: Extracts unique filenames from document metadata
- **Usage**: Populates file selection dropdowns

### load_file_chunks
- **Status**: Active
- **Purpose**: Loads all chunks for a specific file
- **Behavior**: 
  - Retrieves chunks with matching source_file metadata
  - Creates organized DataFrame
  - Includes error handling
- **Usage**: Used in power interface for file viewing

### export_selected_chunks
- **Status**: Active
- **Purpose**: Creates CSV export of selected chunks
- **Behavior**: 
  - Generates timestamped filename
  - Creates temporary file
  - Formats data for Excel compatibility
- **Usage**: Used in both interfaces for data export

### handle_export_with_notification
- **Status**: Active
- **Purpose**: Manages export process with user feedback
- **Behavior**: Shows success/failure notifications during export
- **Usage**: Export button click handler

### handle_file_clear
- **Status**: Active
- **Purpose**: Manages file component visibility
- **Behavior**: Updates UI component visibility state
- **Usage**: Clear handler for export components

## Metadata Management Functions

### add_metadata
- **Status**: Active
- **Purpose**: Adds metadata to selected chunks
- **Behavior**: 
  - Updates ChromaDB metadata
  - Handles special fileset format
  - Refreshes UI view
- **Usage**: Used in both interfaces for metadata management

### delete_metadata
- **Status**: Active
- **Purpose**: Removes metadata from selected chunks
- **Behavior**: 
  - Handles single and multi-value metadata
  - Uses direct SQLite access when needed
  - Updates UI after deletion
- **Usage**: Used in both interfaces for metadata management

## Selection Management Functions

### update_selection_state
- **Status**: Active
- **Purpose**: Updates visual selection state in DataFrame
- **Behavior**: Manages 'Selected' column values
- **Usage**: Called by multiple selection handlers

### handle_select_all
- **Status**: Active
- **Purpose**: Selects all rows in current view
- **Behavior**: Updates selection state for all rows
- **Usage**: Used in both interfaces

### handle_clear_selection
- **Status**: Active
- **Purpose**: Clears all selections
- **Behavior**: Resets selection state to empty
- **Usage**: Used in both interfaces

### handle_select
- **Status**: Active
- **Purpose**: Handles individual row selection
- **Behavior**: Toggles selection state for clicked rows
- **Usage**: DataFrame click handler

### delete_entries
- **Status**: Active
- **Purpose**: Deletes selected entries from database
- **Behavior**: 
  - Removes entries from ChromaDB
  - Updates view after deletion
  - If the entries/chunks deleted are the last chunks in a collection, emptying the collection, then a few additional steps are taken to prevent conflicts that can arise due to orphaned HNSW segment directories. The collection is deleted, the HNSW segment directory is deleted, the collection is recreated with the same name. Then, when a user adds chunks again to this collection, a new HNSW segment directory is created automatically. 
- **Usage**: Used in both interfaces

## Document Upload Functions

### process_file
- **Status**: Active
- **Purpose**: Handles file upload and processing
- **Behavior**: 
  - Sends file to Langflow pipeline
  - Monitors for new chunks
  - Adds metadata to chunks
  - Includes extensive logging
- **Usage**: Core file upload functionality

### update_fileset_inputs
- **Status**: Active
- **Purpose**: Manages fileset input UI
- **Behavior**: Controls visibility of fileset input elements
- **Usage**: Used in document upload interface

### handle_file_upload
- **Status**: Active
- **Purpose**: Top-level upload handler
- **Behavior**: 
  - Manages fileset assignment
  - Coordinates upload process
- **Usage**: Main upload interface handler

## Chat Interface Functions

### handle_chat_with_selection
- **Status**: Active
- **Purpose**: Manages chat interactions
- **Behavior**: 
  - Handles chat requests to Langflow
  - Includes request tracking
  - Manages chat history
- **Usage**: Core chat functionality

## Utility Functions

### show_toast
- **Status**: Active
- **Purpose**: Shows user notifications
- **Behavior**: Converts messages to appropriate toast types
- **Usage**: Used throughout for user feedback

### check_collection_for_files
- **Status**: Active
- **Purpose**: Validates collection content
- **Behavior**: Checks for presence of files in collection
- **Usage**: Collection validation

### handle_collection_file_check
- **Status**: Active
- **Purpose**: UI wrapper for collection checking
- **Behavior**: Manages collection validation feedback
- **Usage**: Collection dropdown handler

### refresh_dropdowns
- **Status**: Active
- **Purpose**: Updates UI dropdown contents
- **Behavior**: Synchronizes dropdown menus with database state
- **Usage**: Used throughout application

## Note
All functions listed above are actively used in the application and have been tested in production scenarios. The documentation is maintained alongside the code and is updated as functionality evolves.
