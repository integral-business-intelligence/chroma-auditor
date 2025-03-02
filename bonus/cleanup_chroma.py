import chromadb
from typing import List, Optional, Tuple, Dict
import logging
import sys
import os
import shutil
import sqlite3

class ChromaCollectionManager:
    def __init__(self, persist_directory: str = "/home/tony/venvs/chroma/chroma_persistent_storage"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_collection_uuid_mapping(self) -> Dict[str, str]:
        """Get mapping between collection names and their UUID directories."""
        try:
            conn = sqlite3.connect(f"{self.persist_directory}/chroma.sqlite3")
            cursor = conn.cursor()
            
            # Query for vector segment IDs associated with collections
            cursor.execute("""
                SELECT c.name, s.id 
                FROM segments s 
                JOIN collections c ON s.collection = c.id 
                WHERE s.scope = 'VECTOR'
            """)
            
            mapping = {name: uuid for name, uuid in cursor.fetchall()}
            conn.close()
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error getting collection-UUID mapping: {e}")
            return {}

    def _get_all_uuid_directories(self) -> List[str]:
        """Get all UUID-like directories in the persistence path."""
        all_dirs = []
        for item in os.listdir(self.persist_directory):
            item_path = os.path.join(self.persist_directory, item)
            # Simple check for UUID-like directories (format checking could be improved)
            if os.path.isdir(item_path) and len(item) >= 32 and "-" in item:
                all_dirs.append(item)
        return all_dirs

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        # Extract just the name strings from the collections
        return [str(col) for col in collections]

    def list_collections_with_uuids(self) -> Tuple[List[Dict], List[str]]:
        """
        List all collections with their UUID directories.
        Also returns a list of orphaned UUID directories.
        """
        collections = self.list_collections()
        uuid_mapping = self._get_collection_uuid_mapping()
        all_uuid_dirs = self._get_all_uuid_directories()
        
        # Create detailed collection info
        collection_info = []
        for col in collections:
            clean_name = str(col).replace('Collection(name=', '').rstrip(')')
            uuid = uuid_mapping.get(clean_name, "Unknown")
            collection_info.append({
                "name": clean_name,
                "uuid": uuid
            })
            
            # Remove this UUID from all_uuid_dirs if it exists
            if uuid in all_uuid_dirs:
                all_uuid_dirs.remove(uuid)
        
        # Any UUIDs left in all_uuid_dirs are orphaned
        return collection_info, all_uuid_dirs

    def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a specific collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            
            # Get UUID from mapping
            uuid_mapping = self._get_collection_uuid_mapping()
            uuid = uuid_mapping.get(collection_name, "Unknown")
            
            return {
                "name": collection_name,
                "count": count,
                "uuid": uuid
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info for {collection_name}: {e}")
            return {
                "name": collection_name,
                "count": 0,
                "uuid": "Unknown",
                "error": str(e)
            }

    def create_collection(self, collection_name: str) -> bool:
        """Create a new empty collection."""
        try:
            self.client.create_collection(name=collection_name)
            self.logger.info(f"Successfully created collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a specific collection including its UUID directory."""
        try:
            # Extract name if it's a Collection object
            name = str(collection_name).replace('Collection(name=', '').rstrip(')')
            
            # Get UUID before deleting collection
            uuid_mapping = self._get_collection_uuid_mapping()
            uuid_dir = uuid_mapping.get(name)
            
            # Delete collection through API
            self.client.delete_collection(name=name)
            self.logger.info(f"Successfully deleted collection metadata: {name}")
            
            # Delete UUID directory if found
            if uuid_dir:
                uuid_path = os.path.join(self.persist_directory, uuid_dir)
                if os.path.exists(uuid_path):
                    shutil.rmtree(uuid_path)
                    self.logger.info(f"Successfully deleted UUID directory: {uuid_dir}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    def delete_collections(self, collection_names: List[str]) -> dict:
        """Delete multiple collections and return results."""
        results = {}
        for name in collection_names:
            # Extract name if it's a Collection object
            clean_name = str(name).replace('Collection(name=', '').rstrip(')')
            results[clean_name] = self.delete_collection(clean_name)
        return results
    
    def delete_orphaned_uuid_dirs(self) -> Tuple[int, List[str]]:
        """Delete orphaned UUID directories."""
        _, orphaned_uuids = self.list_collections_with_uuids()
        deleted = []
        
        for uuid_dir in orphaned_uuids:
            try:
                uuid_path = os.path.join(self.persist_directory, uuid_dir)
                shutil.rmtree(uuid_path)
                deleted.append(uuid_dir)
                self.logger.info(f"Deleted orphaned UUID directory: {uuid_dir}")
            except Exception as e:
                self.logger.error(f"Error deleting orphaned UUID directory {uuid_dir}: {e}")
        
        return len(deleted), deleted

def main():
    manager = ChromaCollectionManager()
    
    while True:
        print("\nChroma Collections Manager")
        print("1. List all collections (with UUIDs)")
        print("2. Get collection info")
        print("3. Delete specific collection")
        print("4. Delete multiple collections")
        print("5. Create new collection")
        print("6. Clean up orphaned UUID directories")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == "1":
            collections, orphaned_uuids = manager.list_collections_with_uuids()
            
            if collections:
                print("\nAvailable collections:")
                for i, col_info in enumerate(collections, 1):
                    print(f"{i}. {col_info['name']} (UUID: {col_info['uuid']})")
            else:
                print("\nNo collections found.")
                
            if orphaned_uuids:
                print(f"\nWARNING: Found {len(orphaned_uuids)} orphaned UUID directories:")
                for uuid_dir in orphaned_uuids:
                    print(f"- {uuid_dir}")
                print("Use option 6 to clean up orphaned directories.")

        elif choice == "2":
            collections = manager.list_collections()
            if not collections:
                print("\nNo collections found.")
                continue
                
            print("\nAvailable collections:")
            for i, name in enumerate(collections, 1):
                clean_name = str(name).replace('Collection(name=', '').rstrip(')')
                print(f"{i}. {clean_name}")
            
            try:
                idx = int(input("\nEnter collection number to inspect: ")) - 1
                if 0 <= idx < len(collections):
                    clean_name = str(collections[idx]).replace('Collection(name=', '').rstrip(')')
                    info = manager.get_collection_info(clean_name)
                    print(f"\nCollection: {info['name']}")
                    print(f"Document count: {info['count']}")
                    print(f"UUID directory: {info['uuid']}")
                    if 'error' in info:
                        print(f"Error: {info['error']}")
                else:
                    print("Invalid collection number.")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "3":
            collections = manager.list_collections()
            if not collections:
                print("\nNo collections found.")
                continue
                
            print("\nAvailable collections:")
            for i, name in enumerate(collections, 1):
                clean_name = str(name).replace('Collection(name=', '').rstrip(')')
                print(f"{i}. {clean_name}")
            
            try:
                idx = int(input("\nEnter collection number to delete: ")) - 1
                if 0 <= idx < len(collections):
                    clean_name = str(collections[idx]).replace('Collection(name=', '').rstrip(')')
                    confirm = input(f"Are you sure you want to delete '{clean_name}'? (y/n): ")
                    if confirm.lower() == 'y':
                        success = manager.delete_collection(clean_name)
                        if success:
                            print(f"Collection '{clean_name}' and its UUID directory deleted successfully.")
                        else:
                            print(f"Failed to delete collection '{clean_name}'.")
                else:
                    print("Invalid collection number.")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "4":
            collections = manager.list_collections()
            if not collections:
                print("\nNo collections found.")
                continue
                
            print("\nAvailable collections:")
            for i, name in enumerate(collections, 1):
                clean_name = str(name).replace('Collection(name=', '').rstrip(')')
                print(f"{i}. {clean_name}")
            
            try:
                indices = input("\nEnter collection numbers to delete (comma-separated): ").split(',')
                indices = [int(idx.strip()) - 1 for idx in indices if idx.strip()]
                
                to_delete = []
                for idx in indices:
                    if 0 <= idx < len(collections):
                        clean_name = str(collections[idx]).replace('Collection(name=', '').rstrip(')')
                        to_delete.append(clean_name)
                    else:
                        print(f"Invalid collection number: {idx + 1}")
                
                if to_delete:
                    print("\nCollections to delete:")
                    for name in to_delete:
                        print(f"- {name}")
                    
                    confirm = input("\nAre you sure you want to delete these collections and their UUID directories? (y/n): ")
                    if confirm.lower() == 'y':
                        results = manager.delete_collections(to_delete)
                        print("\nDeletion results:")
                        for name, success in results.items():
                            status = "Success" if success else "Failed"
                            print(f"- {name}: {status}")
            except ValueError:
                print("Please enter valid numbers.")

        elif choice == "5":
            name = input("\nEnter name for new collection: ").strip()
            if name:
                success = manager.create_collection(name)
                if success:
                    print(f"\nCollection '{name}' created successfully.")
                else:
                    print(f"\nFailed to create collection '{name}'.")
            else:
                print("\nCollection name cannot be empty.")
                
        elif choice == "6":
            print("\nScanning for orphaned UUID directories...")
            count, deleted = manager.delete_orphaned_uuid_dirs()
            
            if count > 0:
                print(f"Successfully deleted {count} orphaned UUID directories:")
                for uuid_dir in deleted:
                    print(f"- {uuid_dir}")
            else:
                print("No orphaned UUID directories found.")

        elif choice == "7":
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
