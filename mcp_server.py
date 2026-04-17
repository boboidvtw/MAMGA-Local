#!/usr/bin/env python3
import sys
import json
import os
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to stderr
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-memory-palace")

# Lazy-loaded system
_system = None

def get_system():
    global _system
    if _system is None:
        from main import TRGSystem
        # We still use TRGSystem to get initialized LLM controller and configs,
        # but we will use the trg_memory instance directly for tools.
        _system = TRGSystem(
            model="gpt-4o-mini",
            embedding_model="minilm",
            cache_dir=str(Path(__file__).parent / "cache")
        )
        try:
            _system.load_memory()
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    return _system

def main():
    tools = [
        {
            "name": "palace_query",
            "description": "Query the Memory Palace (MAMGA) for long-term information based on temporal, semantic, and causal relationships.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the memory system"}
                },
                "required": ["question"]
            }
        },
        {
            "name": "palace_remember",
            "description": "Store a new conversation snippet into the Memory Palace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The conversation turn or text to remember"},
                    "role": {"type": "string", "description": "The role of the speaker (user/assistant)"}
                },
                "required": ["text"]
            }
        }
    ]

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line)
            method = request.get("method")
            req_id = request.get("id")
            
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "MAMGA-Palace", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": tools}
                }
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                
                system = get_system()
                trg = system.trg_memory
                
                result_content = ""
                if tool_name == "palace_query":
                    question = args.get("question")
                    # Use the direct TRG query method to avoid main.py wrapper issues
                    query_context = trg.query(question)
                    result_content = f"Memory Palace Response: {query_context.narrative_context}"
                elif tool_name == "palace_remember":
                    text = args.get("text")
                    role = args.get("role", "user")
                    # Use the direct add_event method
                    trg.add_event(
                        interaction_content=text,
                        timestamp=datetime.now(),
                        metadata={"role": role}
                    )
                    system.save_memory()
                    result_content = "Successfully stored in Memory Palace."
                
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_content}]
                    }
                }
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
                
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error in MCP loop: {e}", exc_info=True)

if __name__ == "__main__":
    main()
