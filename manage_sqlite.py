import sqlite3
import json
import os

DB_PATH = "chat_history.db"

def connect():
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: {DB_PATH} not found. Run main.py first to create it.")
        return None
    return sqlite3.connect(DB_PATH)

def list_sessions():
    conn = connect()
    if not conn: return
    cursor = conn.cursor()
    
    # Fetch all sessions, even ghost ones, so we can manage them
    cursor.execute('''
        SELECT s.session_id, s.title, COUNT(m.id) as msg_count, MAX(m.timestamp)
        FROM sessions s
        LEFT JOIN messages m ON s.session_id = m.session_id
        GROUP BY s.session_id
        ORDER BY MAX(m.timestamp) DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    print("\n" + "="*80)
    print(f"{'SESSION ID':<12} | {'MSGS':<5} | {'LAST ACTIVE':<20} | {'TITLE'}")
    print("-" * 80)
    for row in rows:
        sess_id, title, msg_count, last_active = row
        last_active_str = last_active[:19].replace("T", " ") if last_active else "No Activity"
        print(f"{sess_id:<12} | {msg_count:<5} | {last_active_str:<20} | {title}")
    print("="*80)

def read_session(session_id):
    conn = connect()
    if not conn: return
    cursor = conn.cursor()
    cursor.execute("SELECT role, message_data, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"⚠️ No messages found for session: {session_id}")
        return

    print(f"\n--- Reading Transcript for Session: {session_id} ---")
    for role, data_str, timestamp in rows:
        msg = json.loads(data_str)
        time_str = timestamp[11:19] # Just the HH:MM:SS
        
        # Format the output beautifully based on the role
        if role == "system":
            print(f"[{time_str}] SYSTEM: <System Prompt Loaded>")
        elif role == "user":
            print(f"[{time_str}] USER: {msg.get('content', '')}")
        elif role == "assistant":
            if "tool_calls" in msg:
                calls = [t['function']['name'] for t in msg['tool_calls']]
                print(f"[{time_str}] AGENT: 🛠️  Requested Tools -> {calls}")
            else:
                print(f"[{time_str}] AGENT: {msg.get('content', '')}")
        elif role == "tool":
            content = str(msg.get('content', ''))
            # Truncate massive tool outputs for readability
            display_content = (content[:150] + "...") if len(content) > 150 else content
            print(f"[{time_str}] TOOL ({msg.get('tool_call_id', 'unknown')[:6]}): {display_content}")
    print("-" * 50)

def delete_session(session_id):
    conn = connect()
    if not conn: return
    cursor = conn.cursor()
    
    # Delete from both tables and track how many rows were affected
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    messages_deleted = cursor.rowcount
    
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    sessions_deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    # Check if we actually deleted anything
    if sessions_deleted > 0 or messages_deleted > 0:
        print(f"✅ Session '{session_id}' permanently deleted from SQLite.")
    else:
        print(f"⚠️ Error: Session '{session_id}' not found. Nothing was deleted.")

def prune_ghost_sessions():
    """Deletes any session that has 0 user messages."""
    conn = connect()
    if not conn: return
    cursor = conn.cursor()
    
    # Find sessions with no user messages
    cursor.execute('''
        SELECT s.session_id FROM sessions s
        LEFT JOIN messages m ON s.session_id = m.session_id AND m.role = 'user'
        GROUP BY s.session_id
        HAVING COUNT(m.id) = 0
    ''')
    ghosts = [row[0] for row in cursor.fetchall()]
    
    if not ghosts:
        print("✨ No ghost sessions found. Database is clean!")
        return
        
    for ghost_id in ghosts:
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (ghost_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (ghost_id,))
    
    conn.commit()
    conn.close()
    print(f"🧹 Cleaned up {len(ghosts)} ghost session(s).")

def main():
    while True:
        print("\n🛠️  SQLite Memory Manager")
        print("1. List all sessions")
        print("2. Read a session transcript")
        print("3. Delete a specific session")
        print("4. Prune ghost sessions (Cleanup)")
        print("5. Exit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            list_sessions()
            
        elif choice == '2':
            list_sessions() # Show the menu first!
            sess_id = input("\nEnter the Session ID to read: ").strip()
            if sess_id:
                read_session(sess_id)
                
        elif choice == '3':
            list_sessions() # Show the menu here too!
            sess_id = input("\nEnter the Session ID to delete: ").strip()
            if sess_id:
                delete_session(sess_id)
                
        elif choice == '4':
            prune_ghost_sessions()
            
        elif choice == '5':
            print("Exiting Memory Manager.")
            break
            
        else:
            print("Invalid choice. Please select a number 1-5.")

if __name__ == "__main__":
    main()