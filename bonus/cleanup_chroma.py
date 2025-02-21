# Simple script to manage collections in Chroma. Input your path to persistent storage.

import chromadb
from typing import List, Optional
import logging
import sys

class ChromaCollectionManager:
    def __init__(self, persist_directory: str = "/home/name/venvs/chroma/chroma_persistent_storage"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        # Extract just the name strings from the collections
        return [str(col) for col in collections]

    def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a specific collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info for {collection_name}: {e}")
            return {
                "name": collection_name,
                "count": 0,
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
        """Delete a specific collection."""
        try:
            # Extract name if it's a Collection object
            name = str(collection_name).replace('Collection(name=', '').rstrip(')')
            self.client.delete_collection(name=name)
            self.logger.info(f"Successfully deleted collection: {name}")
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

def main():
    manager = ChromaCollectionManager()
    
    while True:
        print("\nChroma Collections Manager")
        print("1. List all collections")
        print("2. Get collection info")
        print("3. Delete specific collection")
        print("4. Delete multiple collections")
        print("5. Create new collection")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            collections = manager.list_collections()
            if collections:
                print("\nAvailable collections:")
                for i, name in enumerate(collections, 1):
                    # Clean the collection name for display
                    clean_name = str(name).replace('Collection(name=', '').rstrip(')')
                    print(f"{i}. {clean_name}")
            else:
                print("\nNo collections found.")

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
                            print(f"Collection '{clean_name}' deleted successfully.")
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
                    
                    confirm = input("\nAre you sure you want to delete these collections? (y/n): ")
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
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
