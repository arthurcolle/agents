import os
import logging
import asyncio
import json
import time
import multiprocessing
import threading
import queue
import signal
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distributed-services")

class ServiceHealth(BaseModel):
    """Health status of a service"""
    status: str = "unknown"  # unknown, healthy, degraded, unhealthy
    last_check: float = Field(default_factory=time.time)
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    message: str = ""

class ServiceRegistry:
    """
    Registry for managing distributed services with health monitoring and fallback
    """
    def __init__(self, redis_url=None, fallback_redis_urls=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_redis_urls = fallback_redis_urls or [
            url.strip() for url in os.getenv("FALLBACK_REDIS_URLS", "").split(",") if url.strip()
        ]
        self.services = {}
        self.redis_client = None
        self.fallback_clients = []
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        self.health_check_task = None
        self.service_health = {}  # Track health status of services
        self._lock = asyncio.Lock()  # For thread safety
        logger.info(f"Service registry initialized with Redis URL: {self.redis_url} and {len(self.fallback_redis_urls)} fallbacks")
    
    async def connect(self):
        """Connect to Redis with fallback support"""
        # Try primary Redis first
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to primary Redis successfully")
            
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Connect to fallback Redis servers
            for url in self.fallback_redis_urls:
                try:
                    client = await redis.from_url(url)
                    await client.ping()
                    self.fallback_clients.append(client)
                    logger.info(f"Connected to fallback Redis at {url}")
                except Exception as e:
                    logger.warning(f"Failed to connect to fallback Redis at {url}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to primary Redis: {e}")
            
            # Try fallbacks if primary fails
            for url in self.fallback_redis_urls:
                try:
                    self.redis_client = await redis.from_url(url)
                    await self.redis_client.ping()
                    logger.info(f"Connected to fallback Redis at {url} as primary")
                    
                    # Start health check task
                    self.health_check_task = asyncio.create_task(self._health_check_loop())
                    return True
                except Exception as fallback_e:
                    logger.error(f"Failed to connect to fallback Redis at {url}: {fallback_e}")
            
            return False
    
    async def _health_check_loop(self):
        """Periodically check health of all services"""
        try:
            while True:
                await self._check_all_services_health()
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_services_health(self):
        """Check health of all registered services"""
        try:
            services = await self.list_services()
            for service in services:
                service_id = service.get("id")
                if service_id:
                    await self._check_service_health(service_id, service)
        except Exception as e:
            logger.error(f"Error checking services health: {e}")
    
    async def _check_service_health(self, service_id: str, service: Dict):
        """Check health of a specific service"""
        if not service.get("url"):
            return
            
        health = self.service_health.get(service_id, ServiceHealth())
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                try:
                    health_url = f"{service['url']}/health"
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health.status = "healthy"
                            health.response_time = time.time() - start_time
                            health.success_count += 1
                            health.consecutive_failures = 0
                            health.message = "Service is healthy"
                        else:
                            health.status = "degraded"
                            health.response_time = time.time() - start_time
                            health.error_count += 1
                            health.consecutive_failures += 1
                            health.message = f"Service returned status {response.status}"
                except Exception as e:
                    health.status = "unhealthy"
                    health.error_count += 1
                    health.consecutive_failures += 1
                    health.message = f"Health check failed: {str(e)}"
                    
                health.last_check = time.time()
                self.service_health[service_id] = health
                
                # Update service status in registry
                if health.consecutive_failures > 3:
                    service["status"] = "offline"
                    await self._update_service(service_id, service)
                elif health.status != "healthy" and service.get("status") == "online":
                    service["status"] = "degraded"
                    await self._update_service(service_id, service)
                elif health.status == "healthy" and service.get("status") != "online":
                    service["status"] = "online"
                    await self._update_service(service_id, service)
                    
        except Exception as e:
            logger.error(f"Error checking health for service {service_id}: {e}")
    
    async def _update_service(self, service_id: str, service_info: Dict):
        """Update service information in Redis"""
        try:
            await self.redis_client.hset(
                "services", 
                service_id, 
                json.dumps(service_info)
            )
            self.services[service_id] = service_info
            return True
        except Exception as e:
            logger.error(f"Failed to update service {service_id}: {e}")
            
            # Try fallback clients
            for client in self.fallback_clients:
                try:
                    await client.hset(
                        "services", 
                        service_id, 
                        json.dumps(service_info)
                    )
                    return True
                except Exception as fallback_e:
                    continue
                    
            return False
    
    async def register_service(self, service_id: str, service_info: Dict):
        """Register a service with the registry"""
        try:
            async with self._lock:
                service_info["last_heartbeat"] = time.time()
                service_info["registered_at"] = time.time()
                
                # Initialize health status
                self.service_health[service_id] = ServiceHealth(status="unknown")
                
                # Store in Redis
                success = await self._update_service(service_id, service_info)
                
                if success:
                    logger.info(f"Registered service: {service_id}")
                    
                    # Publish service registration event
                    await self._publish_event("service_registered", {
                        "service_id": service_id,
                        "timestamp": time.time()
                    })
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to register service {service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str):
        """Unregister a service from the registry"""
        try:
            async with self._lock:
                # Remove from Redis
                await self.redis_client.hdel("services", service_id)
                
                # Try fallbacks if primary fails
                for client in self.fallback_clients:
                    try:
                        await client.hdel("services", service_id)
                    except Exception:
                        pass
                
                # Remove from local cache
                if service_id in self.services:
                    del self.services[service_id]
                
                # Remove health status
                if service_id in self.service_health:
                    del self.service_health[service_id]
                
                logger.info(f"Unregistered service: {service_id}")
                
                # Publish service unregistration event
                await self._publish_event("service_unregistered", {
                    "service_id": service_id,
                    "timestamp": time.time()
                })
                
                return True
        except Exception as e:
            logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    async def _publish_event(self, event_type: str, data: Dict):
        """Publish an event to Redis pubsub"""
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": time.time()
            }
            await self.redis_client.publish("service_events", json.dumps(event))
            return True
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            
            # Try fallbacks
            for client in self.fallback_clients:
                try:
                    await client.publish("service_events", json.dumps(event))
                    return True
                except Exception:
                    continue
                    
            return False
    
    async def get_service(self, service_id: str) -> Optional[Dict]:
        """Get service information by ID with fallback support"""
        try:
            # Try primary Redis
            service_data = await self.redis_client.hget("services", service_id)
            if service_data:
                service = json.loads(service_data)
                # Add health information if available
                if service_id in self.service_health:
                    service["health"] = self.service_health[service_id].dict()
                return service
                
            # Try fallbacks if primary fails
            for client in self.fallback_clients:
                try:
                    service_data = await client.hget("services", service_id)
                    if service_data:
                        service = json.loads(service_data)
                        # Add health information if available
                        if service_id in self.service_health:
                            service["health"] = self.service_health[service_id].dict()
                        return service
                except Exception:
                    continue
                    
            return None
        except Exception as e:
            logger.error(f"Failed to get service {service_id}: {e}")
            return None
    
    async def list_services(self) -> List[Dict]:
        """List all registered services with fallback support"""
        try:
            # Try primary Redis
            services_data = await self.redis_client.hgetall("services")
            if services_data:
                services = [json.loads(v) for v in services_data.values()]
                # Add health information
                for service in services:
                    if service.get("id") in self.service_health:
                        service["health"] = self.service_health[service.get("id")].dict()
                return services
                
            # Try fallbacks if primary fails
            for client in self.fallback_clients:
                try:
                    services_data = await client.hgetall("services")
                    if services_data:
                        services = [json.loads(v) for v in services_data.values()]
                        # Add health information
                        for service in services:
                            if service.get("id") in self.service_health:
                                service["health"] = self.service_health[service.get("id")].dict()
                        return services
                except Exception:
                    continue
                    
            return []
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return []
    
    async def heartbeat(self, service_id: str):
        """Update service heartbeat with fallback support"""
        try:
            service_data = await self.get_service(service_id)
            if service_data:
                service_data["last_heartbeat"] = time.time()
                
                # Update in Redis
                success = await self._update_service(service_id, service_data)
                
                # Also update health status
                if service_id in self.service_health:
                    health = self.service_health[service_id]
                    health.last_check = time.time()
                    if health.status == "unknown":
                        health.status = "healthy"
                    self.service_health[service_id] = health
                
                return success
            return False
        except Exception as e:
            logger.error(f"Failed to update heartbeat for service {service_id}: {e}")
            return False
    
    async def cleanup_stale_services(self, max_age_seconds=60):
        """Remove services that haven't sent a heartbeat recently"""
        try:
            async with self._lock:
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
    
    async def get_healthy_service(self, service_type: str = None, capability: str = None) -> Optional[Dict]:
        """Get a healthy service of a specific type or with a specific capability"""
        try:
            services = await self.list_services()
            
            # Filter by type or capability if specified
            if service_type:
                services = [s for s in services if s.get("type") == service_type]
            if capability:
                services = [s for s in services if capability in s.get("capabilities", [])]
            
            # Filter by health status
            healthy_services = [s for s in services if s.get("status") == "online" or s.get("status") == "degraded"]
            
            if not healthy_services:
                return None
                
            # Prioritize healthy services
            best_services = [s for s in healthy_services if s.get("status") == "online"]
            
            # If no fully healthy services, use degraded ones
            if not best_services:
                best_services = healthy_services
                
            # Return a random service to distribute load
            return random.choice(best_services)
        except Exception as e:
            logger.error(f"Failed to get healthy service: {e}")
            return None

class ServiceClient:
    """
    Client for interacting with distributed services with automatic fallback
    """
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_connect=5, sock_read=15)
        self.retry_count = int(os.getenv("SERVICE_RETRY_COUNT", "3"))
        self.retry_delay = float(os.getenv("SERVICE_RETRY_DELAY", "1.0"))
        self.circuit_breaker = {}  # Track circuit breaker state for services
        self._lock = asyncio.Lock()  # For thread safety
        logger.info("Service client initialized")
    
    async def connect(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        logger.info("HTTP session initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("HTTP session closed")
    
    async def _check_circuit_breaker(self, service_id: str) -> bool:
        """Check if circuit breaker is open for a service"""
        async with self._lock:
            if service_id in self.circuit_breaker:
                breaker = self.circuit_breaker[service_id]
                if breaker["status"] == "open":
                    # Check if timeout has elapsed
                    if time.time() - breaker["opened_at"] > breaker["timeout"]:
                        # Reset to half-open state
                        breaker["status"] = "half-open"
                        breaker["failures"] = 0
                        logger.info(f"Circuit breaker for service {service_id} reset to half-open")
                        return False
                    return True
            return False
    
    async def _record_success(self, service_id: str):
        """Record a successful call to a service"""
        async with self._lock:
            if service_id in self.circuit_breaker:
                breaker = self.circuit_breaker[service_id]
                if breaker["status"] == "half-open":
                    # Reset circuit breaker on successful call in half-open state
                    breaker["status"] = "closed"
                    breaker["failures"] = 0
                    logger.info(f"Circuit breaker for service {service_id} closed after successful call")
                elif breaker["status"] == "closed":
                    # Reset failure count
                    breaker["failures"] = 0
    
    async def _record_failure(self, service_id: str):
        """Record a failed call to a service"""
        async with self._lock:
            if service_id not in self.circuit_breaker:
                self.circuit_breaker[service_id] = {
                    "status": "closed",
                    "failures": 0,
                    "threshold": 5,  # Open after 5 consecutive failures
                    "timeout": 60,   # Try again after 60 seconds
                    "opened_at": 0
                }
            
            breaker = self.circuit_breaker[service_id]
            if breaker["status"] == "closed" or breaker["status"] == "half-open":
                breaker["failures"] += 1
                
                # Check if threshold reached
                if breaker["failures"] >= breaker["threshold"]:
                    breaker["status"] = "open"
                    breaker["opened_at"] = time.time()
                    logger.warning(f"Circuit breaker for service {service_id} opened after {breaker['failures']} failures")
    
    async def call_service(self, service_id: str, endpoint: str, method="GET", data=None, params=None, 
                          timeout=None, retry_count=None, use_fallback=True) -> Dict:
        """Call a service endpoint with retry and circuit breaker"""
        # Check circuit breaker
        if await self._check_circuit_breaker(service_id):
            return {"success": False, "error": f"Circuit breaker open for service {service_id}"}
        
        # Use provided values or defaults
        retry_count = retry_count if retry_count is not None else self.retry_count
        
        # Get service info
        service = await self.registry.get_service(service_id)
        if not service:
            if use_fallback:
                # Try to find a service with similar capabilities
                if "type" in service_id:  # If service_id contains type info
                    fallback = await self.registry.get_healthy_service(service_type=service_id)
                    if fallback:
                        logger.info(f"Using fallback service {fallback['id']} for {service_id}")
                        return await self.call_service(fallback["id"], endpoint, method, data, params, 
                                                     timeout, retry_count, use_fallback=False)
            return {"success": False, "error": f"Service {service_id} not found"}
        
        # Create session if needed
        if not self.session:
            await self.connect()
        
        # Set custom timeout if provided
        request_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else self.timeout
        
        # Prepare URL
        url = f"{service['url']}/{endpoint.lstrip('/')}"
        
        # Try with retries
        for attempt in range(retry_count + 1):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url, params=params, timeout=request_timeout) as response:
                        result = await response.json()
                        if response.status == 200:
                            await self._record_success(service_id)
                            return {"success": True, "data": result}
                        else:
                            await self._record_failure(service_id)
                            
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data, timeout=request_timeout) as response:
                        result = await response.json()
                        if response.status == 200:
                            await self._record_success(service_id)
                            return {"success": True, "data": result}
                        else:
                            await self._record_failure(service_id)
                            
                elif method.upper() == "PUT":
                    async with self.session.put(url, json=data, timeout=request_timeout) as response:
                        result = await response.json()
                        if response.status == 200:
                            await self._record_success(service_id)
                            return {"success": True, "data": result}
                        else:
                            await self._record_failure(service_id)
                            
                elif method.upper() == "DELETE":
                    async with self.session.delete(url, params=params, timeout=request_timeout) as response:
                        result = await response.json()
                        if response.status == 200:
                            await self._record_success(service_id)
                            return {"success": True, "data": result}
                        else:
                            await self._record_failure(service_id)
                            
                else:
                    return {"success": False, "error": f"Unsupported method: {method}"}
                
                # If we get here, the request failed but didn't raise an exception
                error_msg = f"Service returned status {response.status}"
                
                # Last attempt or non-retryable status code
                if attempt == retry_count or response.status < 500:
                    return {"success": False, "error": error_msg, "status": response.status}
                
                # Exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning(f"Retrying {service_id} after {delay:.2f}s (attempt {attempt+1}/{retry_count})")
                await asyncio.sleep(delay)
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await self._record_failure(service_id)
                
                # Last attempt
                if attempt == retry_count:
                    return {"success": False, "error": f"Request failed after {retry_count} retries: {str(e)}"}
                
                # Exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt) * (0.5 + random.random())
                logger.warning(f"Retrying {service_id} after {delay:.2f}s (attempt {attempt+1}/{retry_count})")
                await asyncio.sleep(delay)
                
            except Exception as e:
                await self._record_failure(service_id)
                logger.error(f"Failed to call service {service_id}: {e}")
                return {"success": False, "error": str(e)}
        
        # Should never reach here, but just in case
        return {"success": False, "error": "Maximum retries exceeded"}
    
    async def call_any_service_with_capability(self, capability: str, endpoint: str, method="GET", 
                                             data=None, params=None) -> Dict:
        """Call any service that has a specific capability"""
        try:
            # Find a healthy service with the required capability
            service = await self.registry.get_healthy_service(capability=capability)
            if not service:
                return {"success": False, "error": f"No healthy service found with capability: {capability}"}
            
            # Call the service
            return await self.call_service(service["id"], endpoint, method, data, params)
        except Exception as e:
            logger.error(f"Failed to call service with capability {capability}: {e}")
            return {"success": False, "error": str(e)}

class TaskQueue:
    """
    Distributed task queue for asynchronous processing with multi-processing support
    """
    def __init__(self, redis_url=None, fallback_redis_urls=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_redis_urls = fallback_redis_urls or [
            url.strip() for url in os.getenv("FALLBACK_REDIS_URLS", "").split(",") if url.strip()
        ]
        self.redis_client = None
        self.fallback_clients = []
        self.task_handlers = {}
        self.process_pool = None
        self.thread_pool = None
        self.max_workers = int(os.getenv("TASK_MAX_WORKERS", str(multiprocessing.cpu_count())))
        self.max_thread_workers = int(os.getenv("TASK_MAX_THREAD_WORKERS", "20"))
        self.pubsub = None
        self.pubsub_task = None
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        logger.info(f"Task queue initialized with Redis URL: {self.redis_url} and {len(self.fallback_redis_urls)} fallbacks")
    
    async def connect(self):
        """Connect to Redis with fallback support and initialize process pool"""
        # Try primary Redis first
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to primary Redis successfully")
            
            # Connect to fallback Redis servers
            for url in self.fallback_redis_urls:
                try:
                    client = await redis.from_url(url)
                    await client.ping()
                    self.fallback_clients.append(client)
                    logger.info(f"Connected to fallback Redis at {url}")
                except Exception as e:
                    logger.warning(f"Failed to connect to fallback Redis at {url}: {e}")
            
            # Initialize process pool for CPU-bound tasks
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            logger.info(f"Initialized process pool with {self.max_workers} workers")
            
            # Initialize thread pool for I/O-bound tasks
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_thread_workers)
            logger.info(f"Initialized thread pool with {self.max_thread_workers} workers")
            
            # Setup PubSub for realtime notifications
            await self._setup_pubsub()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to primary Redis: {e}")
            
            # Try fallbacks if primary fails
            for url in self.fallback_redis_urls:
                try:
                    self.redis_client = await redis.from_url(url)
                    await self.redis_client.ping()
                    logger.info(f"Connected to fallback Redis at {url} as primary")
                    
                    # Initialize process pool for CPU-bound tasks
                    self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
                    logger.info(f"Initialized process pool with {self.max_workers} workers")
                    
                    # Initialize thread pool for I/O-bound tasks
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.max_thread_workers)
                    logger.info(f"Initialized thread pool with {self.max_thread_workers} workers")
                    
                    # Setup PubSub for realtime notifications
                    await self._setup_pubsub()
                    
                    return True
                except Exception as fallback_e:
                    logger.error(f"Failed to connect to fallback Redis at {url}: {fallback_e}")
            
            return False
    
    async def _setup_pubsub(self):
        """Setup Redis PubSub for realtime task notifications"""
        try:
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe("task_events")
            
            # Start pubsub listener task
            self.pubsub_task = asyncio.create_task(self._pubsub_listener())
            logger.info("PubSub listener started")
        except Exception as e:
            logger.error(f"Failed to setup PubSub: {e}")
    
    async def _pubsub_listener(self):
        """Listen for PubSub messages"""
        try:
            while not self._shutdown_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        logger.debug(f"Received PubSub message: {data}")
                        
                        # Handle different event types
                        if data.get("type") == "task_completed":
                            task_id = data.get("data", {}).get("task_id")
                            if task_id:
                                logger.info(f"Task {task_id} completed via PubSub notification")
                                
                        elif data.get("type") == "task_failed":
                            task_id = data.get("data", {}).get("task_id")
                            if task_id:
                                logger.warning(f"Task {task_id} failed via PubSub notification")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in PubSub message: {message['data']}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("PubSub listener cancelled")
        except Exception as e:
            logger.error(f"Error in PubSub listener: {e}")
    
    async def shutdown(self):
        """Shutdown the task queue and release resources"""
        logger.info("Shutting down task queue...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel PubSub task
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass
        
        # Unsubscribe from PubSub
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        # Shutdown process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("Process pool shut down")
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shut down")
        
        logger.info("Task queue shutdown complete")
    
    def register_task_handler(self, task_type: str, handler: Callable, cpu_bound: bool = False):
        """
        Register a handler for a specific task type
        
        Args:
            task_type: Type of task this handler processes
            handler: Function that processes the task
            cpu_bound: Whether this is a CPU-bound task (True) or I/O-bound (False)
        """
        self.task_handlers[task_type] = {
            "handler": handler,
            "cpu_bound": cpu_bound
        }
        logger.info(f"Registered handler for task type: {task_type} (CPU-bound: {cpu_bound})")
    
    async def enqueue_task(self, task_type: str, task_data: Dict, queue_name="default", priority=0):
        """
        Add a task to the queue with priority support
        
        Args:
            task_type: Type of task
            task_data: Data for the task
            queue_name: Name of the queue
            priority: Priority (0-9, higher is more important)
        
        Returns:
            Task ID if successful, None otherwise
        """
        try:
            async with self._lock:
                task_id = f"task_{int(time.time() * 1000)}_{task_type}_{uuid.uuid4().hex[:8]}"
                task = {
                    "id": task_id,
                    "type": task_type,
                    "data": task_data,
                    "status": "pending",
                    "priority": min(9, max(0, priority)),  # Ensure priority is 0-9
                    "created_at": time.time(),
                    "queue": queue_name
                }
                
                # Store task in Redis
                await self._store_task(task)
                
                # Add to appropriate queue based on priority
                if priority > 0:
                    # High priority tasks go to priority queue
                    await self.redis_client.zadd(
                        f"task_priority_queue:{queue_name}",
                        {task_id: 10 - priority}  # Lower score = higher priority
                    )
                else:
                    # Regular tasks go to normal queue
                    await self.redis_client.lpush(
                        f"task_queue:{queue_name}",
                        task_id
                    )
                
                # Publish task enqueued event
                await self._publish_event("task_enqueued", {
                    "task_id": task_id,
                    "task_type": task_type,
                    "queue": queue_name,
                    "priority": priority
                })
                
                logger.info(f"Enqueued task {task_id} of type {task_type} with priority {priority}")
                return task_id
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            return None
    
    async def _store_task(self, task: Dict):
        """Store task data in Redis"""
        try:
            # Store in Redis
            await self.redis_client.hset(
                "tasks",
                task["id"],
                json.dumps(task)
            )
            
            # Try fallbacks if primary fails
            for client in self.fallback_clients:
                try:
                    await client.hset(
                        "tasks",
                        task["id"],
                        json.dumps(task)
                    )
                except Exception:
                    pass
                    
            return True
        except Exception as e:
            logger.error(f"Failed to store task {task['id']}: {e}")
            
            # Try fallbacks
            for client in self.fallback_clients:
                try:
                    await client.hset(
                        "tasks",
                        task["id"],
                        json.dumps(task)
                    )
                    return True
                except Exception:
                    continue
                    
            return False
    
    async def _publish_event(self, event_type: str, data: Dict):
        """Publish an event to Redis pubsub"""
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": time.time()
            }
            await self.redis_client.publish("task_events", json.dumps(event))
            
            # Also publish to specific task channel if task_id is present
            if "task_id" in data:
                await self.redis_client.publish(
                    f"task_events:{data['task_id']}", 
                    json.dumps(event)
                )
                
            return True
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            
            # Try fallbacks
            for client in self.fallback_clients:
                try:
                    await client.publish("task_events", json.dumps(event))
                    
                    # Also publish to specific task channel if task_id is present
                    if "task_id" in data:
                        await client.publish(
                            f"task_events:{data['task_id']}", 
                            json.dumps(event)
                        )
                        
                    return True
                except Exception:
                    continue
                    
            return False
    
    async def _run_in_process_pool(self, handler, task_data):
        """Run a task in the process pool"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool,
            handler,
            task_data
        )
    
    async def _run_in_thread_pool(self, handler, task_data):
        """Run a task in the thread pool"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            handler,
            task_data
        )
    
    async def process_tasks(self, queue_name="default", timeout=0, worker_id=None):
        """
        Process tasks from the queue with multi-processing support
        
        Args:
            queue_name: Name of the queue to process
            timeout: Timeout in seconds (0 = no timeout)
            worker_id: Optional worker ID for logging
        """
        worker_name = f"Worker-{worker_id}" if worker_id else f"Worker-{os.getpid()}-{threading.get_ident()}"
        logger.info(f"{worker_name} started processing tasks from queue {queue_name}")
        
        try:
            while not self._shutdown_event.is_set():
                # First check priority queue
                task_id = None
                
                # Get highest priority task
                priority_result = await self.redis_client.zrange(
                    f"task_priority_queue:{queue_name}",
                    0, 0  # Get the lowest score (highest priority)
                )
                
                if priority_result:
                    # Remove from priority queue
                    task_id = priority_result[0]
                    await self.redis_client.zrem(
                        f"task_priority_queue:{queue_name}",
                        task_id
                    )
                else:
                    # No priority tasks, check regular queue
                    result = await self.redis_client.brpop(
                        f"task_queue:{queue_name}",
                        timeout=1 if timeout == 0 else min(timeout, 1)  # Use short timeout to check shutdown event
                    )
                    
                    if not result:
                        if timeout > 0:
                            # Decrement timeout
                            timeout -= 1
                            if timeout <= 0:
                                logger.info(f"{worker_name}: No tasks in queue {queue_name} after waiting")
                                break
                        continue
                    
                    _, task_id = result
                
                # Get task data
                task_data = await self.redis_client.hget("tasks", task_id)
                if not task_data:
                    logger.warning(f"{worker_name}: Task {task_id} not found in storage")
                    continue
                    
                task = json.loads(task_data)
                
                # Update task status
                task["status"] = "processing"
                task["started_at"] = time.time()
                task["worker"] = worker_name
                
                # Store updated task state
                await self._store_task(task)
                
                # Publish task started event
                await self._publish_event("task_started", {
                    "task_id": task["id"],
                    "worker": worker_name
                })
                
                # Process task
                task_type = task["type"]
                if task_type in self.task_handlers:
                    handler_info = self.task_handlers[task_type]
                    handler = handler_info["handler"]
                    cpu_bound = handler_info["cpu_bound"]
                    
                    try:
                        logger.info(f"{worker_name}: Processing task {task['id']} of type {task_type}")
                        
                        # Run in appropriate pool based on task type
                        if cpu_bound:
                            # CPU-bound task goes to process pool
                            result = await self._run_in_process_pool(handler, task["data"])
                        else:
                            # I/O-bound task goes to thread pool
                            result = await self._run_in_thread_pool(handler, task["data"])
                        
                        # Update task with result
                        task["status"] = "completed"
                        task["result"] = result
                        task["completed_at"] = time.time()
                        
                        # Publish task completion event
                        await self._publish_event("task_completed", {
                            "task_id": task["id"],
                            "worker": worker_name
                        })
                        
                    except Exception as e:
                        logger.error(f"{worker_name}: Error processing task {task['id']}: {e}")
                        task["status"] = "failed"
                        task["error"] = str(e)
                        task["failed_at"] = time.time()
                        
                        # Publish task failure event
                        await self._publish_event("task_failed", {
                            "task_id": task["id"],
                            "error": str(e),
                            "worker": worker_name
                        })
                else:
                    logger.warning(f"{worker_name}: No handler for task type {task_type}")
                    task["status"] = "failed"
                    task["error"] = f"No handler for task type {task_type}"
                    task["failed_at"] = time.time()
                    
                    # Publish task failure event
                    await self._publish_event("task_failed", {
                        "task_id": task["id"],
                        "error": f"No handler for task type {task_type}",
                        "worker": worker_name
                    })
                
                # Update task in storage
                await self._store_task(task)
                
        except asyncio.CancelledError:
            logger.info(f"{worker_name}: Task processing cancelled")
        except Exception as e:
            logger.error(f"{worker_name}: Error in task processing loop: {e}")
    
    async def start_workers(self, queue_name="default", num_workers=None):
        """
        Start multiple worker tasks to process tasks from the queue
        
        Args:
            queue_name: Name of the queue to process
            num_workers: Number of workers to start (defaults to CPU count)
        
        Returns:
            List of worker tasks
        """
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            
        logger.info(f"Starting {num_workers} workers for queue {queue_name}")
        
        workers = []
        for i in range(num_workers):
            worker = asyncio.create_task(self.process_tasks(queue_name=queue_name, worker_id=i+1))
            workers.append(worker)
            
        return workers
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID with fallback support"""
        try:
            # Try primary Redis
            task_data = await self.redis_client.hget("tasks", task_id)
            if task_data:
                return json.loads(task_data)
                
            # Try fallbacks if primary fails
            for client in self.fallback_clients:
                try:
                    task_data = await client.hget("tasks", task_id)
                    if task_data:
                        return json.loads(task_data)
                except Exception:
                    continue
                    
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    async def wait_for_task(self, task_id: str, timeout=60) -> Optional[Dict]:
        """Wait for a task to complete with improved reliability"""
        try:
            # Check if task already completed
            task = await self.get_task(task_id)
            if task and task["status"] in ["completed", "failed"]:
                return task
            
            # Create a future to be resolved when task completes
            loop = asyncio.get_running_loop()
            completion_future = loop.create_future()
            
            # Subscribe to task events
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(f"task_events:{task_id}")
            
            # Start listener task
            async def listen_for_completion():
                try:
                    while not completion_future.done():
                        message = await pubsub.get_message(timeout=1)
                        if message and message["type"] == "message":
                            try:
                                event = json.loads(message["data"])
                                event_type = event.get("type")
                                event_data = event.get("data", {})
                                
                                if (event_type in ["task_completed", "task_failed"] and 
                                    event_data.get("task_id") == task_id):
                                    # Task completed or failed, resolve future
                                    if not completion_future.done():
                                        completion_future.set_result(True)
                                        break
                            except json.JSONDecodeError:
                                logger.error(f"Invalid JSON in task event: {message['data']}")
                        
                        # Periodically check task status directly as a fallback
                        # This handles cases where we might miss the pubsub message
                        if random.random() < 0.1:  # 10% chance each iteration
                            direct_check = await self.get_task(task_id)
                            if direct_check and direct_check["status"] in ["completed", "failed"]:
                                if not completion_future.done():
                                    completion_future.set_result(True)
                                    break
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in task completion listener: {e}")
                    if not completion_future.done():
                        completion_future.set_exception(e)
                finally:
                    await pubsub.unsubscribe(f"task_events:{task_id}")
            
            # Start listener
            listener_task = asyncio.create_task(listen_for_completion())
            
            try:
                # Wait for completion or timeout
                await asyncio.wait_for(completion_future, timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for task {task_id}")
            finally:
                # Clean up
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
            
            # Get final task state
            return await self.get_task(task_id)
        except Exception as e:
            logger.error(f"Failed to wait for task {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task"""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False
                
            # Can only cancel pending tasks
            if task["status"] not in ["pending", "processing"]:
                return False
                
            # Update task status
            task["status"] = "cancelled"
            task["cancelled_at"] = time.time()
            
            # Store updated task
            await self._store_task(task)
            
            # Publish cancellation event
            await self._publish_event("task_cancelled", {
                "task_id": task_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def retry_task(self, task_id: str, priority=None) -> Optional[str]:
        """Retry a failed task"""
        try:
            task = await self.get_task(task_id)
            if not task:
                return None
                
            # Can only retry failed or cancelled tasks
            if task["status"] not in ["failed", "cancelled"]:
                return None
                
            # Create new task with same data
            new_task_id = await self.enqueue_task(
                task["type"],
                task["data"],
                queue_name=task.get("queue", "default"),
                priority=priority if priority is not None else task.get("priority", 0)
            )
            
            # Update original task to show it was retried
            task["retried"] = True
            task["retry_task_id"] = new_task_id
            await self._store_task(task)
            
            return new_task_id
        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {e}")
            return None
    
    async def list_tasks(self, status=None, task_type=None, limit=100) -> List[Dict]:
        """List tasks with optional filtering"""
        try:
            # Get all task IDs
            task_ids = await self.redis_client.hkeys("tasks")
            
            # Get task data for each ID
            tasks = []
            for task_id in task_ids[-limit:]:  # Limit to most recent tasks
                task_data = await self.redis_client.hget("tasks", task_id)
                if task_data:
                    task = json.loads(task_data)
                    
                    # Apply filters
                    if status and task.get("status") != status:
                        continue
                    if task_type and task.get("type") != task_type:
                        continue
                        
                    tasks.append(task)
                    
                    # Stop if we've reached the limit
                    if len(tasks) >= limit:
                        break
            
            return tasks
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []

class RealtimePubSub:
    """
    Realtime PubSub for distributed communication
    """
    def __init__(self, redis_url=None, fallback_redis_urls=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_redis_urls = fallback_redis_urls or [
            url.strip() for url in os.getenv("FALLBACK_REDIS_URLS", "").split(",") if url.strip()
        ]
        self.redis_client = None
        self.fallback_clients = []
        self.pubsub = None
        self.subscribers = {}  # channel -> list of callbacks
        self.listener_task = None
        self._shutdown_event = asyncio.Event()
        logger.info(f"Realtime PubSub initialized with Redis URL: {self.redis_url}")
    
    async def connect(self):
        """Connect to Redis with fallback support"""
        # Try primary Redis first
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to primary Redis successfully for PubSub")
            
            # Connect to fallback Redis servers
            for url in self.fallback_redis_urls:
                try:
                    client = await redis.from_url(url)
                    await client.ping()
                    self.fallback_clients.append(client)
                    logger.info(f"Connected to fallback Redis at {url} for PubSub")
                except Exception as e:
                    logger.warning(f"Failed to connect to fallback Redis at {url} for PubSub: {e}")
            
            # Initialize PubSub
            self.pubsub = self.redis_client.pubsub()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to primary Redis for PubSub: {e}")
            
            # Try fallbacks if primary fails
            for url in self.fallback_redis_urls:
                try:
                    self.redis_client = await redis.from_url(url)
                    await self.redis_client.ping()
                    logger.info(f"Connected to fallback Redis at {url} as primary for PubSub")
                    
                    # Initialize PubSub
                    self.pubsub = self.redis_client.pubsub()
                    
                    return True
                except Exception as fallback_e:
                    logger.error(f"Failed to connect to fallback Redis at {url} for PubSub: {fallback_e}")
            
            return False
    
    async def subscribe(self, channel: str, callback: Callable):
        """Subscribe to a channel with a callback"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
            # Subscribe to the channel in Redis
            await self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
        
        # Add callback
        self.subscribers[channel].append(callback)
        
        # Start listener if not already running
        if not self.listener_task or self.listener_task.done():
            self.listener_task = asyncio.create_task(self._message_listener())
    
    async def unsubscribe(self, channel: str, callback: Callable = None):
        """Unsubscribe from a channel"""
        if channel not in self.subscribers:
            return
            
        if callback:
            # Remove specific callback
            self.subscribers[channel] = [cb for cb in self.subscribers[channel] if cb != callback]
            
            # If no more callbacks, unsubscribe from channel
            if not self.subscribers[channel]:
                await self.pubsub.unsubscribe(channel)
                del self.subscribers[channel]
                logger.info(f"Unsubscribed from channel: {channel}")
        else:
            # Remove all callbacks
            await self.pubsub.unsubscribe(channel)
            del self.subscribers[channel]
            logger.info(f"Unsubscribed from channel: {channel}")
    
    async def publish(self, channel: str, message: Any):
        """Publish a message to a channel"""
        try:
            # Convert message to JSON if it's not a string
            if not isinstance(message, str):
                message = json.dumps(message)
                
            # Publish to Redis
            await self.redis_client.publish(channel, message)
            logger.debug(f"Published message to channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            
            # Try fallbacks
            for client in self.fallback_clients:
                try:
                    await client.publish(channel, message)
                    logger.debug(f"Published message to channel {channel} via fallback")
                    return True
                except Exception:
                    continue
                    
            return False
    
    async def _message_listener(self):
        """Listen for messages and dispatch to callbacks"""
        try:
            logger.info("Started PubSub message listener")
            while not self._shutdown_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    
                    # Parse JSON if possible
                    try:
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        if isinstance(data, str):
                            data = json.loads(data)
                    except json.JSONDecodeError:
                        # Keep as is if not valid JSON
                        pass
                    
                    # Dispatch to callbacks
                    if channel in self.subscribers:
                        for callback in self.subscribers[channel]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    asyncio.create_task(callback(channel, data))
                                else:
                                    callback(channel, data)
                            except Exception as e:
                                logger.error(f"Error in PubSub callback: {e}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("PubSub message listener cancelled")
        except Exception as e:
            logger.error(f"Error in PubSub message listener: {e}")
    
    async def shutdown(self):
        """Shutdown the PubSub and release resources"""
        logger.info("Shutting down PubSub...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel listener task
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        # Unsubscribe from all channels
        if self.pubsub:
            channels = list(self.subscribers.keys())
            if channels:
                await self.pubsub.unsubscribe(*channels)
            await self.pubsub.close()
        
        logger.info("PubSub shutdown complete")

class BaseService:
    """
    Base class for distributed services with health monitoring
    """
    def __init__(self, service_id: str, registry: ServiceRegistry, host="localhost", port=8001, 
                service_type="generic", capabilities=None):
        self.service_id = service_id
        self.registry = registry
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.service_type = service_type
        self.capabilities = capabilities or []
        self.heartbeat_task = None
        self.health_check_task = None
        self.pubsub = None
        self.status = "initializing"
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_response_time": 0,
            "uptime_start": time.time()
        }
        self._shutdown_event = asyncio.Event()
        logger.info(f"{service_type.capitalize()} service initialized with ID: {service_id}")
    
    async def start(self):
        """Start the service"""
        # Connect to registry
        await self.registry.connect()
        
        # Initialize PubSub
        self.pubsub = RealtimePubSub()
        await self.pubsub.connect()
        
        # Subscribe to service events
        await self.pubsub.subscribe("service_events", self._handle_service_event)
        
        # Register service
        service_info = {
            "id": self.service_id,
            "name": f"{self.service_type.capitalize()} Service",
            "type": self.service_type,
            "description": f"A {self.service_type} service",
            "url": self.url,
            "capabilities": self.capabilities,
            "status": "online",
            "version": "1.0.0",
            "metrics": self.metrics
        }
        
        await self.registry.register_service(self.service_id, service_info)
        self.status = "online"
        
        # Start heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start health check
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Service {self.service_id} started")
    
    async def stop(self):
        """Stop the service"""
        logger.info(f"Stopping service {self.service_id}...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Update status
        self.status = "shutting_down"
        
        # Cancel heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Cancel health check
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Unregister service
        await self.registry.unregister_service(self.service_id)
        
        # Shutdown PubSub
        if self.pubsub:
            await self.pubsub.shutdown()
        
        logger.info(f"Service {self.service_id} stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the registry"""
        try:
            while not self._shutdown_event.is_set():
                # Update metrics
                self.metrics["uptime"] = time.time() - self.metrics["uptime_start"]
                
                # Send heartbeat
                await self.registry.heartbeat(self.service_id)
                
                # Wait for next heartbeat
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        except asyncio.CancelledError:
            logger.info(f"Heartbeat loop for service {self.service_id} cancelled")
        except Exception as e:
            logger.error(f"Error in heartbeat loop for service {self.service_id}: {e}")
    
    async def _health_check_loop(self):
        """Perform periodic health checks"""
        try:
            while not self._shutdown_event.is_set():
                # Perform health check
                health_status = await self._check_health()
                
                # Update status based on health
                if health_status["status"] != self.status:
                    self.status = health_status["status"]
                    
                    # Update service info in registry
                    service = await self.registry.get_service(self.service_id)
                    if service:
                        service["status"] = self.status
                        service["metrics"] = self.metrics
                        await self.registry.register_service(self.service_id, service)
                
                # Wait for next check
                await asyncio.sleep(60)  # Check health every 60 seconds
        except asyncio.CancelledError:
            logger.info(f"Health check loop for service {self.service_id} cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop for service {self.service_id}: {e}")
    
    async def _check_health(self):
        """
        Check health of the service
        Override in subclasses to implement specific health checks
        """
        return {
            "status": "online",
            "message": "Service is healthy"
        }
    
    async def _handle_service_event(self, channel, data):
        """Handle service events from PubSub"""
        try:
            if isinstance(data, dict):
                event_type = data.get("type")
                event_data = data.get("data", {})
                
                if event_type == "service_registered":
                    service_id = event_data.get("service_id")
                    if service_id != self.service_id:
                        logger.info(f"New service registered: {service_id}")
                
                elif event_type == "service_unregistered":
                    service_id = event_data.get("service_id")
                    if service_id != self.service_id:
                        logger.info(f"Service unregistered: {service_id}")
        except Exception as e:
            logger.error(f"Error handling service event: {e}")

# Example service implementation
class ExampleService(BaseService):
    """
    Example service implementation with enhanced capabilities
    """
    def __init__(self, service_id: str, registry: ServiceRegistry, host="localhost", port=8001):
        super().__init__(
            service_id=service_id,
            registry=registry,
            host=host,
            port=port,
            service_type="example",
            capabilities=["data_processing", "text_analysis"]
        )
        self.task_queue = None
    
    async def start(self):
        """Start the service with task queue"""
        # Initialize task queue
        self.task_queue = TaskQueue()
        await self.task_queue.connect()
        
        # Register task handlers
        self.task_queue.register_task_handler(
            "process_data", 
            self._process_data_task,
            cpu_bound=True  # CPU-bound task
        )
        
        self.task_queue.register_task_handler(
            "analyze_text", 
            self._analyze_text_task,
            cpu_bound=False  # I/O-bound task
        )
        
        # Start task workers
        self.workers = await self.task_queue.start_workers(num_workers=2)
        
        # Start the base service
        await super().start()
    
    async def stop(self):
        """Stop the service and task queue"""
        # Cancel workers
        for worker in getattr(self, 'workers', []):
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        
        # Shutdown task queue
        if self.task_queue:
            await self.task_queue.shutdown()
        
        # Stop the base service
        await super().stop()
    
    async def _check_health(self):
        """Implement health check for example service"""
        # Check if task queue is operational
        if not self.task_queue or not self.task_queue.redis_client:
            return {
                "status": "degraded",
                "message": "Task queue not connected"
            }
        
        # Check if workers are running
        active_workers = sum(1 for w in getattr(self, 'workers', []) if not w.done())
        if active_workers < len(getattr(self, 'workers', [])):
            return {
                "status": "degraded",
                "message": f"Only {active_workers} of {len(getattr(self, 'workers', []))} workers active"
            }
        
        return {
            "status": "online",
            "message": "Service is healthy"
        }
    
    async def _process_data_task(self, data):
        """Example CPU-bound task handler"""
        logger.info(f"Processing data task: {data}")
        
        # Simulate CPU-intensive processing
        result = {"processed": True}
        
        # Process input data
        if "input" in data:
            # Simulate complex processing
            time.sleep(1)  # Simulate work
            result["output"] = f"Processed: {data['input']}"
            
        return result
    
    async def _analyze_text_task(self, data):
        """Example I/O-bound task handler"""
        logger.info(f"Analyzing text task: {data}")
        
        # Simulate I/O-bound processing
        await asyncio.sleep(0.5)  # Simulate I/O wait
        
        result = {"analyzed": True}
        
        # Analyze text
        if "text" in data:
            text = data["text"]
            result["word_count"] = len(text.split())
            result["char_count"] = len(text)
            result["sentiment"] = "positive" if "good" in text.lower() else "neutral"
            
        return result

class WebRTCService(BaseService):
    """
    WebRTC service for peer-to-peer communication
    """
    def __init__(self, service_id: str, registry: ServiceRegistry, host="localhost", port=8002,
                signaling_port=8003):
        super().__init__(
            service_id=service_id,
            registry=registry,
            host=host,
            port=port,
            service_type="webrtc",
            capabilities=["peer_communication", "media_streaming", "data_channel"]
        )
        self.signaling_port = signaling_port
        self.signaling_url = f"ws://{host}:{signaling_port}"
        self.peers = {}
        self.rooms = {}
        self.signaling_server = None
    
    async def start(self):
        """Start the WebRTC service with signaling server"""
        # Start the base service
        await super().start()
        
        # Update service info with signaling URL
        service = await self.registry.get_service(self.service_id)
        if service:
            service["signaling_url"] = self.signaling_url
            await self.registry.register_service(self.service_id, service)
        
        # Subscribe to WebRTC-specific events
        await self.pubsub.subscribe(f"webrtc:{self.service_id}", self._handle_webrtc_event)
        
        logger.info(f"WebRTC service started with signaling at {self.signaling_url}")
    
    async def _handle_webrtc_event(self, channel, data):
        """Handle WebRTC-specific events"""
        try:
            if isinstance(data, dict):
                event_type = data.get("type")
                
                if event_type == "peer_connected":
                    peer_id = data.get("peer_id")
                    logger.info(f"Peer connected: {peer_id}")
                    
                elif event_type == "peer_disconnected":
                    peer_id = data.get("peer_id")
                    logger.info(f"Peer disconnected: {peer_id}")
                    
                elif event_type == "room_created":
                    room_id = data.get("room_id")
                    logger.info(f"Room created: {room_id}")
        except Exception as e:
            logger.error(f"Error handling WebRTC event: {e}")
    
    async def create_room(self, room_id=None):
        """Create a new room for WebRTC peers"""
        if not room_id:
            room_id = f"room_{uuid.uuid4().hex[:8]}"
            
        self.rooms[room_id] = {
            "id": room_id,
            "created_at": time.time(),
            "peers": [],
            "active": True
        }
        
        # Publish room creation event
        await self.pubsub.publish(f"webrtc:{self.service_id}", {
            "type": "room_created",
            "room_id": room_id,
            "timestamp": time.time()
        })
        
        return room_id
    
    async def join_room(self, room_id, peer_id):
        """Add a peer to a room"""
        if room_id not in self.rooms:
            return False
            
        room = self.rooms[room_id]
        if peer_id not in room["peers"]:
            room["peers"].append(peer_id)
            
            # Publish peer joined event
            await self.pubsub.publish(f"webrtc:{self.service_id}", {
                "type": "peer_joined",
                "room_id": room_id,
                "peer_id": peer_id,
                "timestamp": time.time()
            })
            
        return True
    
    async def leave_room(self, room_id, peer_id):
        """Remove a peer from a room"""
        if room_id not in self.rooms:
            return False
            
        room = self.rooms[room_id]
        if peer_id in room["peers"]:
            room["peers"].remove(peer_id)
            
            # Publish peer left event
            await self.pubsub.publish(f"webrtc:{self.service_id}", {
                "type": "peer_left",
                "room_id": room_id,
                "peer_id": peer_id,
                "timestamp": time.time()
            })
            
            # If room is empty, mark as inactive
            if not room["peers"]:
                room["active"] = False
                
        return True
    
    async def get_room_peers(self, room_id):
        """Get all peers in a room"""
        if room_id not in self.rooms:
            return []
            
        return self.rooms[room_id]["peers"]
    
    async def _check_health(self):
        """Implement health check for WebRTC service"""
        # Check if signaling server is running
        if not self.signaling_server:
            return {
                "status": "degraded",
                "message": "Signaling server not running"
            }
            
        return {
            "status": "online",
            "message": "Service is healthy"
        }

# Example usage with multi-processing
async def example():
    # Initialize registry with fallback support
    registry = ServiceRegistry(
        fallback_redis_urls=["redis://backup:6379/0"]
    )
    await registry.connect()
    
    # Initialize task queue with multi-processing
    task_queue = TaskQueue()
    await task_queue.connect()
    
    # Register task handlers for different types of tasks
    def cpu_bound_task(data):
        # This runs in a separate process
        logger.info(f"Processing CPU-bound task: {data}")
        # Simulate CPU-intensive work
        result = 0
        for i in range(1000000):
            result += i
        return {"processed": True, "result": result}
    
    async def io_bound_task(data):
        # This runs in a thread
        logger.info(f"Processing I/O-bound task: {data}")
        # Simulate I/O operations
        await asyncio.sleep(1)
        return {"processed": True, "result": f"Processed {data['input']}"}
    
    # Register handlers specifying CPU or I/O bound
    task_queue.register_task_handler("cpu_task", cpu_bound_task, cpu_bound=True)
    task_queue.register_task_handler("io_task", io_bound_task, cpu_bound=False)
    
    # Start multiple workers
    workers = await task_queue.start_workers(num_workers=4)
    
    # Enqueue tasks with different priorities
    tasks = []
    for i in range(5):
        # CPU-bound tasks
        task_id = await task_queue.enqueue_task(
            "cpu_task", 
            {"input": f"cpu_data_{i}"},
            priority=i % 3  # Different priorities
        )
        tasks.append(task_id)
        
        # I/O-bound tasks
        task_id = await task_queue.enqueue_task(
            "io_task", 
            {"input": f"io_data_{i}"},
            priority=i % 3  # Different priorities
        )
        tasks.append(task_id)
    
    # Setup realtime PubSub
    pubsub = RealtimePubSub()
    await pubsub.connect()
    
    # Subscribe to task events
    async def handle_task_event(channel, data):
        logger.info(f"Received task event: {data}")
    
    await pubsub.subscribe("task_events", handle_task_event)
    
    # Wait for all tasks to complete
    results = []
    for task_id in tasks:
        result = await task_queue.wait_for_task(task_id)
        results.append(result)
    
    logger.info(f"All tasks completed: {len(results)} results")
    
    # Start a WebRTC service
    webrtc_service = WebRTCService("webrtc-1", registry)
    await webrtc_service.start()
    
    # Create a room
    room_id = await webrtc_service.create_room()
    logger.info(f"Created WebRTC room: {room_id}")
    
    # Clean up
    for worker in workers:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    
    await task_queue.shutdown()
    await pubsub.shutdown()
    await webrtc_service.stop()

if __name__ == "__main__":
    asyncio.run(example())
