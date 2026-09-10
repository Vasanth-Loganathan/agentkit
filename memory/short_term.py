import sqlite3
import json
import uuid
import datetime
import threading
import copy
import os
from utils.base64encoder import encode_image

class ShortTermMemory:
    """Manages context with Episodic SQLite persistence, a Sticky Anchor, and LLM Auto-Naming."""

    def __init__(self, llm_client, db_path="chat_history.db", max_tokens=32000):
        self.llm_client = llm_client
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.messages = []
        self.session_id = None
        
        self._init_db()

    def _init_db(self):
        """Creates the tables for both sessions and messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Stores the individual messages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                message_data TEXT,
                timestamp DATETIME
            )
        ''')
        
        # Table 2: Stores the Session Metadata (for our beautiful menu)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME
            )
        ''')
        conn.commit()
        conn.close()

    def load_session(self, session_id=None):
        if session_id is None:
            self.session_id = str(uuid.uuid4())[:8] 
            self.messages = []
            
            # The Magic Flag: We haven't touched the DB yet!
            self.session_started_in_db = False 
        else:
            self.session_id = session_id
            self.session_started_in_db = True
            self._fetch_history()

    def _fetch_history(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT message_data FROM messages WHERE session_id = ? ORDER BY id ASC', (self.session_id,))
        rows = cursor.fetchall()
        conn.close()

        self.messages = []
        for row in rows:
            self.messages.append(json.loads(row[0]))

    def _generate_title(self, first_user_message: str):
            """Runs purely in the background to force an LLM-generated title."""
            print(f"\n[DEBUG] 🔄 Auto-namer thread started for session {self.session_id}...")
            try:
                prompt = f"Write a descriptive 3 to 5 word title for a chat that starts with this prompt. Only return the title, no quotes. Prompt: {first_user_message}"
                title_messages = [{"role": "user", "content": prompt}]
                
                response = self.llm_client.generate(messages=title_messages, tools=None)
                new_title = response.content.strip('"\'\n ')
                    
                # Open a fresh SQLite connection for the thread to avoid locks
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE sessions SET title = ? WHERE session_id = ?', (new_title, self.session_id))
                conn.commit()
                conn.close()
                                
            except Exception as e:
                pass

    def add_message(self, message: dict):
        # Always add to RAM so the active Agent Loop functions normally
        self.messages.append(message)

        if not self.session_id:
            return

        db_message = copy.deepcopy(message)
        
        if isinstance(db_message.get('content'), list):
            for block in db_message['content']:
                if block.get('type') == 'image_url':
                    # Drop the massive Base64 string and save the local file path
                    safe_path = db_message.get('local_path', 'unknown_image.png')
                    block['image_url']['url'] = f"local_file:{safe_path}"
                    
        # Remove the temporary key so it doesn't get saved to the DB
        db_message.pop('local_path', None) 

        if not getattr(self, 'session_started_in_db', True):
            if message.get('role') != 'user':
                return # It's just the System Prompt. Keep it in RAM and wait.
                
            # The user finally spoke! Let's lock everything into the database.
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Register the Session
            cursor.execute(
                'INSERT INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)',
                (self.session_id, "New Chat", datetime.datetime.now().isoformat())
            )
            
            # 2. Batch-save everything currently in RAM 
            for msg in self.messages:
                save_msg = copy.deepcopy(msg)
                if isinstance(save_msg.get('content'), list):
                    for block in save_msg['content']:
                        if block.get('type') == 'image_url':
                            block['image_url']['url'] = f"local_file:{save_msg.get('local_path', 'img')}"
                save_msg.pop('local_path', None)
                
                cursor.execute(
                    'INSERT INTO messages (session_id, role, message_data, timestamp) VALUES (?, ?, ?, ?)',
                    (self.session_id, save_msg.get('role', 'unknown'), json.dumps(save_msg), datetime.datetime.now().isoformat())
                )
            conn.commit()
            conn.close()
            
            self.session_started_in_db = True
            
            # 3. Trigger the Auto-namer thread
            content = message.get('content')
            text_only = next((item['text'] for item in content if item.get('type') == 'text'), "Image upload") if isinstance(content, list) else content
            
            threading.Thread(target=self._generate_title, args=(text_only,)).start()
            
        else:
            # Standard operation for a session that is already fully active in the DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (session_id, role, message_data, timestamp) VALUES (?, ?, ?, ?)',
                (self.session_id, db_message.get('role', 'unknown'), json.dumps(db_message), datetime.datetime.now().isoformat())
            )
            conn.commit()
            conn.close()

    def _estimate_tokens(self, msg: dict) -> int:
        """Smarter token estimation that doesn't count raw Base64 characters."""
        import copy
        temp_msg = copy.deepcopy(msg)
        image_count = 0
        
        # If it's a multimodal list, find the image and strip the base64 just for counting
        if isinstance(temp_msg.get('content'), list):
            for block in temp_msg['content']:
                if block.get('type') == 'image_url':
                    block['image_url']['url'] = "" 
                    image_count += 1
                    
        # Calculate standard text tokens + a flat 1000 tokens per image
        base_tokens = len(json.dumps(temp_msg)) // 4
        return base_tokens + (image_count * 1000)

    def _rehydrate_message(self, msg: dict) -> dict:
        """Converts local_file: paths back to base64 data URIs for API ingestion."""
        import copy
        msg_copy = copy.deepcopy(msg)
        content = msg_copy.get("content")

        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("local_file:"):
                        file_path = url.replace("local_file:", "")
                        
                        if os.path.exists(file_path):
                            # Use the imported utility function to get the Base64 string
                            b64_data = encode_image(file_path)
                            block["image_url"]["url"] = f"data:image/jpeg;base64,{b64_data}"
                        else:
                            # Fallback if image was moved or deleted
                            block["type"] = "text"
                            block["text"] = f"[Image file {os.path.basename(file_path)} not found on disk]"
                            block.pop("image_url", None)
                            
        return msg_copy
    
    def get_messages(self):
        """Sticky Anchor + Sliding Window with Image Rehydration."""
        if not self.messages:
            return []

        remaining_budget = self.max_tokens
        anchors, recent_window = [], []
        start_idx = 0

        # Anchor 1: System Prompt
        if len(self.messages) > 0 and self.messages[0].get('role') == 'system':
            anchors.append(self.messages[0])
            remaining_budget -= self._estimate_tokens(self.messages[0])
            start_idx = 1

        # Anchor 2: First User Message
        if len(self.messages) > start_idx and self.messages[start_idx].get('role') == 'user':
            anchors.append(self.messages[start_idx])
            remaining_budget -= self._estimate_tokens(self.messages[start_idx])
            start_idx += 1

        # Sliding Window for Recent Messages
        for msg in reversed(self.messages[start_idx:]):
            estimated_tokens = self._estimate_tokens(msg)
            if remaining_budget - estimated_tokens < 0:
                break
            recent_window.insert(0, msg)
            remaining_budget -= estimated_tokens

        # Rehydrate any local_file paths in the active context window
        raw_context = anchors + recent_window
        return [self._rehydrate_message(m) for m in raw_context]
    

    def get_all_sessions(self):
        """Fetches sessions but filters out Ghost Sessions using HAVING."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # The HAVING clause ensures we ONLY show sessions where the user actually spoke
        cursor.execute('''
            SELECT s.session_id, s.title, COUNT(m.id) as msg_count, MAX(m.timestamp) as last_active 
            FROM sessions s
            JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id 
            HAVING SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) > 0
            ORDER BY last_active DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        # Format timestamps nicely
        formatted_rows = []
        for sess_id, title, msg_count, last_active in rows:
            time_str = "No Activity"
            if last_active:
                dt = datetime.datetime.fromisoformat(last_active)
                time_str = dt.strftime("%b %d, %I:%M %p")
            formatted_rows.append((sess_id, msg_count, time_str, title))
            
        return formatted_rows