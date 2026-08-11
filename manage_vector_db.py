import os

import hashlib
import json
import lancedb

def manual_db_control():
    persist_dir = "./lancedb_data"
    table_name = "knowledge_base"
    
    if not os.path.exists(persist_dir):
        print("❌ Database directory not found. Run your main agent first to initialize it.")
        return

    db = lancedb.connect(persist_dir)
    
    try:
        table = db.open_table(table_name)
    except Exception:
        print("❌ Table not found. Run your main agent first to initialize it.")
        return

    while True:
        print("\n================ VECTOR DB MANAGER ================")
        print("1. View all records (Read)")
        print("2. Add a new record (Create)")
        print("3. Update a record (Update)")
        print("4. Delete a record (Delete)")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            data = table.to_arrow().to_pylist()
            print(f"\n📦 Total Records: {len(data)}")
            for record in data:
                print(f"ID: {record['id']} | Text: {record['text'][:75]}...")
                
        elif choice == '2':
            new_text = input("\nEnter the text for the new document: ").strip()
            if new_text:
                doc_id = f"doc_{hashlib.md5(new_text.encode('utf-8')).hexdigest()}"
                
                # No manual embedding required! The table schema handles it automatically.
                new_record = [{
                    "id": doc_id,
                    "text": new_text,
                    "metadata": json.dumps({"source": "manual_cli_entry"})
                }]
                
                try:
                    table.add(new_record)
                    print(f"✅ Successfully added new document with ID: '{doc_id}'.")
                except Exception as e:
                    print(f"❌ Failed to add document: {e}")
                    
        elif choice == '3':
            target_id = input("\nEnter the exact ID of the document to UPDATE (or 'cancel'): ").strip()
            if target_id.lower() != 'cancel':
                updated_text = input("Enter the new text for this document: ").strip()
                if updated_text:
                    try:
                        table.delete(f"id = '{target_id}'")
                        
                        updated_record = [{
                            "id": target_id,
                            "text": updated_text,
                            "metadata": json.dumps({"source": "manual_cli_update"})
                        }]
                        
                        table.add(updated_record)
                        print(f"✅ Successfully updated document '{target_id}'.")
                    except Exception as e:
                        print(f"❌ Failed to update document: {e}")
        
        elif choice == '4':
            target_id = input("\nEnter the exact ID of the document to DELETE (or type 'cancel'): ").strip()
            if target_id.lower() != 'cancel':
                try:
                    table.delete(f"id = '{target_id}'")
                    print(f"✅ Successfully deleted '{target_id}'.")
                except Exception as e:
                    print(f"❌ Failed to delete: {e}")
                    
        elif choice == '5':
            print("Exiting manager...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    manual_db_control()