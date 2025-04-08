import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Import from openai_agent with error handling
try:
    from openai_agent import (
        AutonomousAgent, TranslationOrchestrator, SYSTEM_GOAL,
        search_web, fact_check, read_url, JINA_AVAILABLE
    )
except ImportError as e:
    logging.error(f"Error importing from openai_agent: {e}")
    # Define fallbacks
    SYSTEM_GOAL = "Build a cool web app with dynamic generative UI"
    JINA_AVAILABLE = False
    
    async def search_web(query: str):
        return {"error": "Search functionality not available"}
    
    async def fact_check(statement: str):
        return {"error": "Fact check functionality not available"}
    
    async def read_url(url: str):
        return {"error": "URL reading functionality not available"}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-app")

# Initialize FastAPI app
app = FastAPI(title="Dynamic Generative UI Agent")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create default HTML template if it doesn't exist
index_html_path = "templates/index.html"
if not os.path.exists(index_html_path):
    with open(index_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Generative UI Agent</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #e1f5fe;
            align-self: flex-end;
            margin-left: auto;
        }
        .agent-message {
            background-color: #f0f0f0;
            align-self: flex-start;
        }
        .input-area {
            display: flex;
            padding: 10px;
            background-color: #fff;
            border-top: 1px solid #ddd;
        }
        #message-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }
        button {
            padding: 10px 15px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0b7dda;
        }
        .dynamic-ui {
            margin-top: 20px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fff;
        }
        .system-goal {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0,0,0,.3);
            border-radius: 50%;
            border-top-color: #2196F3;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>Dynamic Generative UI Agent</h1>
    
    <div class="system-goal">
        <h3>System Goal:</h3>
        <p id="system-goal-text">Loading system goal...</p>
    </div>
    
    <div class="chat-container">
        <div class="chat-messages" id="chat-messages">
            <div class="message agent-message">
                Hello! I'm your assistant. I can help you build a web app with dynamic generative UI.
                What would you like to work on today?
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="message-input" placeholder="Type your message here...">
            <button id="send-button">Send</button>
        </div>
    </div>
    
    <div class="dynamic-ui" id="dynamic-ui">
        <h2>Dynamic UI Area</h2>
        <p>This area will update with dynamically generated UI elements based on your conversation.</p>
    </div>
    
    <script>
        const chatMessages = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const dynamicUI = document.getElementById('dynamic-ui');
        const systemGoalText = document.getElementById('system-goal-text');
        
        // Fetch system goal
        fetch('/api/system-goal')
            .then(response => response.json())
            .then(data => {
                systemGoalText.textContent = data.goal;
            });
        
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = function(e) {
            console.log('WebSocket connection established');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'message') {
                addMessage(data.content, 'agent');
            } else if (data.type === 'ui_update') {
                updateDynamicUI(data.content);
            } else if (data.type === 'thinking') {
                showThinking(data.content);
            }
        };
        
        ws.onclose = function(event) {
            if (event.wasClean) {
                console.log(`Connection closed cleanly, code=${event.code}, reason=${event.reason}`);
            } else {
                console.log('Connection died');
                addMessage('Connection to server lost. Please refresh the page.', 'agent');
            }
        };
        
        ws.onerror = function(error) {
            console.log(`WebSocket error: ${error.message}`);
        };
        
        // Send message
        function sendMessage() {
            const message = messageInput.value.trim();
            if (message) {
                addMessage(message, 'user');
                ws.send(JSON.stringify({
                    type: 'message',
                    content: message
                }));
                messageInput.value = '';
            }
        }
        
        // Add message to chat
        function addMessage(content, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = content;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Show thinking indicator
        function showThinking(message) {
            const thinkingDiv = document.createElement('div');
            thinkingDiv.className = 'message agent-message thinking';
            thinkingDiv.innerHTML = message + ' <span class="loading"></span>';
            thinkingDiv.id = 'thinking-indicator';
            chatMessages.appendChild(thinkingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Update dynamic UI area
        function updateDynamicUI(htmlContent) {
            dynamicUI.innerHTML = htmlContent;
            
            // Execute any scripts in the dynamic UI
            const scripts = dynamicUI.querySelectorAll('script');
            scripts.forEach(script => {
                const newScript = document.createElement('script');
                Array.from(script.attributes).forEach(attr => {
                    newScript.setAttribute(attr.name, attr.value);
                });
                newScript.textContent = script.textContent;
                script.parentNode.replaceChild(newScript, script);
            });
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
        """)
    logger.info(f"Created default HTML template at {index_html_path}")

# Set up static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.autonomous_agent = AutonomousAgent(name="WebUI Assistant", model="gpt-4")
        self.autonomous_agent.load_context()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def send_message(self, websocket: WebSocket, message: str):
        await websocket.send_json({
            "type": "message",
            "content": message
        })
        
    async def send_thinking(self, websocket: WebSocket, message: str):
        await websocket.send_json({
            "type": "thinking",
            "content": message
        })
        
    async def send_ui_update(self, websocket: WebSocket, html_content: str):
        await websocket.send_json({
            "type": "ui_update",
            "content": html_content
        })
        
    def _format_search_results_html(self, results: Dict[str, Any], query: str) -> str:
        """Format search results as HTML"""
        if results.get("error"):
            return f"""
            <div class="error-container">
                <h3>Error searching for "{query}"</h3>
                <p>{results.get("message", "Unknown error")}</p>
                <p>Make sure you have set the JINA_API_KEY environment variable.</p>
            </div>
            """
        
        if results.get("mock", False):
            mock_notice = """
            <div class="notice" style="background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin-bottom: 15px;">
                <p><strong>Note:</strong> These are mock results. For real results, please set the JINA_API_KEY environment variable.</p>
            </div>
            """
        else:
            mock_notice = ""
        
        result_items = ""
        search_results = results.get("results", [])
        
        if isinstance(search_results, list):
            for item in search_results:
                title = item.get("title", "No title")
                url = item.get("url", "#")
                snippet = item.get("snippet", "No description available")
                
                result_items += f"""
                <div class="search-result">
                    <h4><a href="{url}" target="_blank">{title}</a></h4>
                    <p class="url">{url}</p>
                    <p class="snippet">{snippet}</p>
                </div>
                """
        else:
            result_items = f"<p>Unexpected result format: {search_results}</p>"
        
        return f"""
        <div class="search-results-container">
            <h3>Search Results for "{query}"</h3>
            {mock_notice}
            <div class="search-results">
                {result_items if result_items else "<p>No results found</p>"}
            </div>
            <style>
                .search-results-container {{
                    font-family: Arial, sans-serif;
                }}
                .search-result {{
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .search-result h4 {{
                    margin-bottom: 5px;
                }}
                .search-result .url {{
                    color: #006621;
                    font-size: 0.8em;
                    margin-bottom: 5px;
                }}
                .search-result .snippet {{
                    color: #545454;
                    font-size: 0.9em;
                }}
            </style>
        </div>
        """
    
    def _format_fact_check_html(self, results: Dict[str, Any], statement: str) -> str:
        """Format fact check results as HTML"""
        if results.get("error"):
            return f"""
            <div class="error-container">
                <h3>Error fact-checking "{statement}"</h3>
                <p>{results.get("message", "Unknown error")}</p>
                <p>Make sure you have set the JINA_API_KEY environment variable.</p>
            </div>
            """
        
        if results.get("mock", False):
            mock_notice = """
            <div class="notice" style="background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin-bottom: 15px;">
                <p><strong>Note:</strong> These are mock results. For real results, please set the JINA_API_KEY environment variable.</p>
            </div>
            """
        else:
            mock_notice = ""
        
        fact_check = results.get("factCheck", {})
        verdict = fact_check.get("verdict", "UNKNOWN")
        confidence = fact_check.get("confidence", 0)
        explanation = fact_check.get("explanation", "No explanation available")
        
        # Determine verdict color
        verdict_color = "#6c757d"  # Default gray
        if verdict == "TRUE":
            verdict_color = "#28a745"  # Green
        elif verdict == "FALSE":
            verdict_color = "#dc3545"  # Red
        elif verdict == "PARTIALLY_TRUE":
            verdict_color = "#fd7e14"  # Orange
        
        # Format sources
        sources_html = ""
        sources = results.get("sources", [])
        if sources:
            sources_items = ""
            for source in sources:
                title = source.get("title", "Unnamed Source")
                url = source.get("url", "#")
                relevance = source.get("relevance", 0)
                
                sources_items += f"""
                <li>
                    <a href="{url}" target="_blank">{title}</a>
                    <span class="relevance">Relevance: {relevance:.2f}</span>
                </li>
                """
            
            sources_html = f"""
            <div class="sources">
                <h4>Sources:</h4>
                <ul>
                    {sources_items}
                </ul>
            </div>
            """
        
        return f"""
        <div class="fact-check-container">
            <h3>Fact Check: "{statement}"</h3>
            {mock_notice}
            <div class="verdict" style="background-color: {verdict_color}; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>Verdict: {verdict}</strong>
                <span class="confidence">Confidence: {confidence:.2f}</span>
            </div>
            <div class="explanation">
                <h4>Explanation:</h4>
                <p>{explanation}</p>
            </div>
            {sources_html}
            <style>
                .fact-check-container {{
                    font-family: Arial, sans-serif;
                }}
                .confidence {{
                    float: right;
                    font-size: 0.9em;
                }}
                .sources {{
                    margin-top: 20px;
                }}
                .sources ul {{
                    padding-left: 20px;
                }}
                .relevance {{
                    font-size: 0.8em;
                    color: #6c757d;
                    margin-left: 10px;
                }}
            </style>
        </div>
        """
    
    def _format_read_url_html(self, results: Dict[str, Any], url: str) -> str:
        """Format URL reading results as HTML"""
        if results.get("error"):
            return f"""
            <div class="error-container">
                <h3>Error reading URL: {url}</h3>
                <p>{results.get("message", "Unknown error")}</p>
                <p>Make sure you have set the JINA_API_KEY environment variable.</p>
            </div>
            """
        
        if results.get("mock", False):
            mock_notice = """
            <div class="notice" style="background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin-bottom: 15px;">
                <p><strong>Note:</strong> These are mock results. For real results, please set the JINA_API_KEY environment variable.</p>
            </div>
            """
        else:
            mock_notice = ""
        
        title = results.get("title", "No title")
        content = results.get("content", "No content available")
        summary = results.get("summary", "No summary available")
        
        return f"""
        <div class="url-content-container">
            <h3>Content from: <a href="{url}" target="_blank">{url}</a></h3>
            {mock_notice}
            <div class="url-title">
                <h4>{title}</h4>
            </div>
            <div class="url-summary">
                <h4>Summary:</h4>
                <p>{summary}</p>
            </div>
            <div class="url-content">
                <h4>Content:</h4>
                <div class="content-box">
                    {content}
                </div>
            </div>
            <style>
                .url-content-container {{
                    font-family: Arial, sans-serif;
                }}
                .url-title {{
                    margin-bottom: 15px;
                }}
                .url-summary {{
                    margin-bottom: 20px;
                }}
                .content-box {{
                    max-height: 300px;
                    overflow-y: auto;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }}
            </style>
        </div>
        """
        
    async def process_message(self, websocket: WebSocket, message: str):
        # Send thinking indicator
        await self.send_thinking(websocket, "Thinking...")
        
        # Check for special commands
        if message.startswith("/search "):
            query = message[8:].strip()
            results = await search_web(query)
            
            # Format results for display
            response = {
                "conversation_response": f"Here are the search results for '{query}':",
                "ui_elements": self._format_search_results_html(results, query)
            }
            
            await self.send_message(websocket, response["conversation_response"])
            await self.send_ui_update(websocket, response["ui_elements"])
            return
            
        elif message.startswith("/fact-check "):
            statement = message[12:].strip()
            results = await fact_check(statement)
            
            # Format results for display
            response = {
                "conversation_response": f"Fact check results for '{statement}':",
                "ui_elements": self._format_fact_check_html(results, statement)
            }
            
            await self.send_message(websocket, response["conversation_response"])
            await self.send_ui_update(websocket, response["ui_elements"])
            return
            
        elif message.startswith("/read-url "):
            url = message[10:].strip()
            results = await read_url(url)
            
            # Format results for display
            response = {
                "conversation_response": f"Content analysis for {url}:",
                "ui_elements": self._format_read_url_html(results, url)
            }
            
            await self.send_message(websocket, response["conversation_response"])
            await self.send_ui_update(websocket, response["ui_elements"])
            return
        
        # Process with autonomous agent
        enhanced_prompt = f"""
        The user is interacting with a web application that has a dynamic UI.
        Based on their request, provide:
        1. A conversational response
        2. Any UI elements that would help them accomplish their task
        
        Available special commands:
        - /search [query]: Search the web
        - /fact-check [statement]: Fact check a statement
        - /read-url [url]: Read and analyze content from a URL
        
        User request: {message}
        
        Format your response as JSON with these fields:
        {{
            "conversation_response": "Your conversational response here",
            "ui_elements": "HTML/CSS/JS code for any UI elements to display"
        }}
        """
        
        result = await self.autonomous_agent.run_async(enhanced_prompt)
        
        # Try to parse the response as JSON
        try:
            # Extract JSON from the response if it's wrapped in markdown code blocks
            if "```json" in result and "```" in result.split("```json", 1)[1]:
                json_str = result.split("```json", 1)[1].split("```", 1)[0].strip()
                response_data = json.loads(json_str)
            elif "```" in result and "```" in result.split("```", 1)[1]:
                json_str = result.split("```", 1)[1].split("```", 1)[0].strip()
                response_data = json.loads(json_str)
            else:
                # Try to parse the whole response as JSON
                response_data = json.loads(result)
                
            # Send the conversation response
            if "conversation_response" in response_data:
                await self.send_message(websocket, response_data["conversation_response"])
            else:
                await self.send_message(websocket, "I processed your request but couldn't generate a proper response.")
                
            # Send UI update if available
            if "ui_elements" in response_data and response_data["ui_elements"]:
                await self.send_ui_update(websocket, response_data["ui_elements"])
                
        except json.JSONDecodeError:
            # If JSON parsing fails, send the raw response
            logger.error(f"Failed to parse JSON response: {result}")
            await self.send_message(websocket, result)

# Initialize connection manager
manager = ConnectionManager()

# API routes
class SystemGoal(BaseModel):
    goal: str

class SearchQuery(BaseModel):
    query: str

class FactCheckQuery(BaseModel):
    statement: str

class ReadUrlQuery(BaseModel):
    url: str

# Create API router
api_router = APIRouter(prefix="/api")

@api_router.get("/system-goal", response_model=SystemGoal)
async def get_system_goal():
    return {"goal": SYSTEM_GOAL}

@api_router.post("/search")
async def api_search(query: SearchQuery):
    results = await search_web(query.query)
    return results

@api_router.post("/fact-check")
async def api_fact_check(query: FactCheckQuery):
    results = await fact_check(query.statement)
    return results

@api_router.post("/read-url")
async def api_read_url(query: ReadUrlQuery):
    results = await read_url(query.url)
    return results

@api_router.get("/jina-status")
async def get_jina_status():
    return {"available": JINA_AVAILABLE}

# Register API router
app.include_router(api_router)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                if json_data.get("type") == "message":
                    await manager.process_message(websocket, json_data.get("content", ""))
                else:
                    await manager.process_message(websocket, data)
            except json.JSONDecodeError:
                await manager.process_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# HTML routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Main function to run the app
def run_app(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_app()
