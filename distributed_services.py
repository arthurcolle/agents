import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distributed-services")

class ServiceRegistry:
    """
    Registry for managing distributed services
    """
    def __init__(self, redis_url=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.services = {}
        self.redis_client = None
        logger.info(f"Service registry initialized with Redis URL: {self.redis_url}")
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def register_service(self, service_id: str, service_info: Dict):
        """Register a service with the registry"""
        try:
            service_info["last_heartbeat"] = time.time()
            await self.redis_client.hset(
                "services", 
                service_id, 
                json.dumps(service_info)
            )
            self.services[service_id] = service_info
            logger.info(f"Registered service: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register service {service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str):
        """Unregister a service from the registry"""
        try:
            await self.redis_client.hdel("services", service_id)
            if service_id in self.services:
                del self.services[service_id]
            logger.info(f"Unregistered service: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    async def get_service(self, service_id: str) -> Optional[Dict]:
        """Get service information by ID"""
        try:
            service_data = await self.redis_client.hget("services", service_id)
            if service_data:
                return json.loads(service_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get service {service_id}: {e}")
            return None
    
    async def list_services(self) -> List[Dict]:
        """List all registered services"""
        try:
            services_data = await self.redis_client.hgetall("services")
            return [json.loads(v) for v in services_data.values()]
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return []
    
    async def heartbeat(self, service_id: str):
        """Update service heartbeat"""
        try:
            service_data = await self.get_service(service_id)
            if service_data:
                service_data["last_heartbeat"] = time.time()
                await self.redis_client.hset(
                    "services", 
                    service_id, 
                    json.dumps(service_data)
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update heartbeat for service {service_id}: {e}")
            return False
    
    async def cleanup_stale_services(self, max_age_seconds=60):
        """Remove services that haven't sent a heartbeat recently"""
        try:
            current_time = time.time()
            services = await self.list_services()
            
            for service in services:
                service_id = service.get("id")
                last_heartbeat = service.get("last_heartbeat", 0)
                
                if current_time - last_heartbeat > max_age_seconds:
                    await self.unregister_service(service_id)
                    logger.info(f"Removed stale service: {service_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup stale services: {e}")
            return False

class ServiceClient:
    """
    Client for interacting with distributed services
    """
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.session = None
        logger.info("Service client initialized")
    
    async def connect(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("HTTP session initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("HTTP session closed")
    
    async def call_service(self, service_id: str, endpoint: str, method="GET", data=None, params=None) -> Dict:
        """Call a service endpoint"""
        try:
            service = await self.registry.get_service(service_id)
            if not service:
                return {"success": False, "error": f"Service {service_id} not found"}
            
            url = f"{service['url']}/{endpoint.lstrip('/')}"
            
            if not self.session:
                await self.connect()
            
            if method.upper() == "GET":
                async with self.session.get(url, params=params) as response:
                    result = await response.json()
                    return {"success": response.status == 200, "data": result}
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
                    return {"success": response.status == 200, "data": result}
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data) as response:
                    result = await response.json()
                    return {"success": response.status == 200, "data": result}
            elif method.upper() == "DELETE":
                async with self.session.delete(url, params=params) as response:
                    result = await response.json()
                    return {"success": response.status == 200, "data": result}
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}
        except Exception as e:
            logger.error(f"Failed to call service {service_id}: {e}")
            return {"success": False, "error": str(e)}

class TaskQueue:
    """
    Distributed task queue for asynchronous processing
    """
    def __init__(self, redis_url=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self.task_handlers = {}
        logger.info(f"Task queue initialized with Redis URL: {self.redis_url}")
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def enqueue_task(self, task_type: str, task_data: Dict, queue_name="default"):
        """Add a task to the queue"""
        try:
            task = {
                "id": f"task_{int(time.time() * 1000)}_{task_type}",
                "type": task_type,
                "data": task_data,
                "status": "pending",
                "created_at": time.time()
            }
            
            await self.redis_client.lpush(
                f"task_queue:{queue_name}",
                json.dumps(task)
            )
            
            logger.info(f"Enqueued task {task['id']} of type {task_type}")
            return task["id"]
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            return None
    
    async def process_tasks(self, queue_name="default", timeout=0):
        """Process tasks from the queue"""
        try:
            while True:
                # Get task from queue with timeout
                result = await self.redis_client.brpop(
                    f"task_queue:{queue_name}",
                    timeout=timeout
                )
                
                if not result:
                    if timeout > 0:
                        logger.info(f"No tasks in queue {queue_name} after waiting {timeout} seconds")
                        break
                    continue
                
                _, task_json = result
                task = json.loads(task_json)
                
                # Update task status
                task["status"] = "processing"
                task["started_at"] = time.time()
                
                # Store task state
                await self.redis_client.hset(
                    "tasks",
                    task["id"],
                    json.dumps(task)
                )
                
                # Process task
                task_type = task["type"]
                if task_type in self.task_handlers:
                    try:
                        logger.info(f"Processing task {task['id']} of type {task_type}")
                        result = await self.task_handlers[task_type](task["data"])
                        
                        # Update task with result
                        task["status"] = "completed"
                        task["result"] = result
                        task["completed_at"] = time.time()
                        
                    except Exception as e:
                        logger.error(f"Error processing task {task['id']}: {e}")
                        task["status"] = "failed"
                        task["error"] = str(e)
                        task["failed_at"] = time.time()
                else:
                    logger.warning(f"No handler for task type {task_type}")
                    task["status"] = "failed"
                    task["error"] = f"No handler for task type {task_type}"
                    task["failed_at"] = time.time()
                
                # Update task in storage
                await self.redis_client.hset(
                    "tasks",
                    task["id"],
                    json.dumps(task)
                )
                
                # Publish task completion event
                await self.redis_client.publish(
                    f"task_events:{task['id']}",
                    json.dumps({"status": task["status"], "task_id": task["id"]})
                )
                
        except asyncio.CancelledError:
            logger.info("Task processing cancelled")
        except Exception as e:
            logger.error(f"Error in task processing loop: {e}")
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID"""
        try:
            task_data = await self.redis_client.hget("tasks", task_id)
            if task_data:
                return json.loads(task_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    async def wait_for_task(self, task_id: str, timeout=60) -> Optional[Dict]:
        """Wait for a task to complete"""
        try:
            # Check if task already completed
            task = await self.get_task(task_id)
            if task and task["status"] in ["completed", "failed"]:
                return task
            
            # Subscribe to task events
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(f"task_events:{task_id}")
            
            # Wait for completion event
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = await pubsub.get_message(timeout=1)
                if message and message["type"] == "message":
                    event = json.loads(message["data"])
                    if event["task_id"] == task_id:
                        await pubsub.unsubscribe(f"task_events:{task_id}")
                        return await self.get_task(task_id)
            
            # Timeout reached
            await pubsub.unsubscribe(f"task_events:{task_id}")
            return await self.get_task(task_id)
        except Exception as e:
            logger.error(f"Failed to wait for task {task_id}: {e}")
            return None

# Example service implementation
class ExampleService:
    """
    Example service implementation
    """
    def __init__(self, service_id: str, registry: ServiceRegistry, host="localhost", port=8001):
        self.service_id = service_id
        self.registry = registry
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.heartbeat_task = None
        logger.info(f"Example service initialized with ID: {service_id}")
    
    async def start(self):
        """Start the service"""
        # Connect to registry
        await self.registry.connect()
        
        # Register service
        service_info = {
            "id": self.service_id,
            "name": "Example Service",
            "description": "An example distributed service",
            "url": self.url,
            "capabilities": ["data_processing", "text_analysis"],
            "status": "online"
        }
        
        await self.registry.register_service(self.service_id, service_info)
        
        # Start heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"Service {self.service_id} started")
    
    async def stop(self):
        """Stop the service"""
        # Cancel heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Unregister service
        await self.registry.unregister_service(self.service_id)
        
        logger.info(f"Service {self.service_id} stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the registry"""
        try:
            while True:
                await self.registry.heartbeat(self.service_id)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        except asyncio.CancelledError:
            logger.info(f"Heartbeat loop for service {self.service_id} cancelled")
        except Exception as e:
            logger.error(f"Error in heartbeat loop for service {self.service_id}: {e}")

# Example usage
async def example():
    # Initialize registry
    registry = ServiceRegistry()
    await registry.connect()
    
    # Initialize task queue
    task_queue = TaskQueue()
    await task_queue.connect()
    
    # Register task handler
    async def process_data_task(data):
        logger.info(f"Processing data task: {data}")
        # Simulate processing
        await asyncio.sleep(2)
        return {"processed": True, "result": f"Processed {data['input']}"}
    
    task_queue.register_task_handler("process_data", process_data_task)
    
    # Start task processing in background
    task_processor = asyncio.create_task(task_queue.process_tasks())
    
    # Enqueue a task
    task_id = await task_queue.enqueue_task(
        "process_data", 
        {"input": "sample data", "options": {"format": "json"}}
    )
    
    # Wait for task completion
    task_result = await task_queue.wait_for_task(task_id)
    logger.info(f"Task result: {task_result}")
    
    # Clean up
    task_processor.cancel()
    try:
        await task_processor
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(example())
