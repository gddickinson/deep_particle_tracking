"""
Thread manager for handling background processing and thread safety.
"""

import threading
import logging
import queue
import concurrent.futures
from typing import Callable, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Task:
    """Represents a background task with associated metadata."""
    
    def __init__(self, 
                 task_id: str, 
                 func: Callable, 
                 args: Tuple = (), 
                 kwargs: Dict = None,
                 callback: Callable = None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.callback = callback
        self.status = "pending"
        self.result = None
        self.error = None

    def execute(self):
        """Execute the task and handle success/failure."""
        try:
            self.status = "running"
            self.result = self.func(*self.args, **self.kwargs)
            self.status = "completed"
            return self.result
        except Exception as e:
            self.status = "failed"
            self.error = e
            logger.error(f"Task {self.task_id} failed: {str(e)}")
            raise e


class ThreadManager:
    """Manages background threads for processing tasks."""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.task_futures = {}
        self.task_lock = threading.Lock()
        self.completed_callbacks = {}
        self._shutdown = False
    
    def submit_task(self, 
                   task_id: str, 
                   func: Callable, 
                   args: Tuple = (), 
                   kwargs: Dict = None, 
                   callback: Callable = None) -> str:
        """Submit a task to be executed in a background thread."""
        if self._shutdown:
            raise RuntimeError("ThreadManager has been shut down")
        
        kwargs = kwargs or {}
        task = Task(task_id, func, args, kwargs, callback)
        
        with self.task_lock:
            self.tasks[task_id] = task
            future = self.executor.submit(task.execute)
            self.task_futures[task_id] = future
            
            if callback:
                future.add_done_callback(
                    lambda f, task_id=task_id: self._handle_task_completion(task_id, f)
                )
        
        logger.debug(f"Submitted task: {task_id}")
        return task_id
    
    def _handle_task_completion(self, task_id: str, future):
        """Handle the completion of a task and execute its callback."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        try:
            result = future.result()  # This will re-raise any exception from the task
            if task.callback:
                task.callback(result)
        except Exception as e:
            logger.error(f"Error in task {task_id}: {str(e)}")
            task.status = "failed"
            task.error = e
    
    def get_task_status(self, task_id: str) -> Dict:
        """Get the current status of a task."""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return {"status": "unknown", "task_id": task_id}
            
            future = self.task_futures.get(task_id)
            if future:
                if future.done():
                    if future.exception():
                        return {
                            "status": "failed",
                            "task_id": task_id,
                            "error": str(future.exception())
                        }
                    else:
                        return {
                            "status": "completed",
                            "task_id": task_id
                        }
                elif future.running():
                    return {"status": "running", "task_id": task_id}
                else:
                    return {"status": "pending", "task_id": task_id}
            
            return {
                "status": task.status,
                "task_id": task_id,
                "error": str(task.error) if task.error else None
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Attempt to cancel a running task."""
        with self.task_lock:
            future = self.task_futures.get(task_id)
            if future and not future.done():
                cancelled = future.cancel()
                if cancelled:
                    task = self.tasks.get(task_id)
                    if task:
                        task.status = "cancelled"
                    logger.debug(f"Cancelled task: {task_id}")
                return cancelled
            return False
    
    def get_result(self, task_id: str, timeout=None) -> Any:
        """Get the result of a completed task."""
        with self.task_lock:
            future = self.task_futures.get(task_id)
            if not future:
                raise KeyError(f"No task found with ID: {task_id}")
        
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Task {task_id} did not complete within the specified timeout")
    
    def wait_for_task(self, task_id: str, timeout=None) -> bool:
        """Wait for a specific task to complete."""
        with self.task_lock:
            future = self.task_futures.get(task_id)
            if not future:
                return False
        
        try:
            future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            return False
        except Exception:
            # Task failed, but we were just waiting for completion
            return True
    
    def shutdown(self, wait=True):
        """Shutdown the thread executor."""
        self._shutdown = True
        self.executor.shutdown(wait=wait)
    
    def get_active_tasks(self) -> List[str]:
        """Get a list of all active task IDs."""
        active_tasks = []
        with self.task_lock:
            for task_id, future in self.task_futures.items():
                if not future.done():
                    active_tasks.append(task_id)
        return active_tasks
    
    def get_task_count(self) -> Dict[str, int]:
        """Get a count of tasks by status."""
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
        
        with self.task_lock:
            for task_id, future in self.task_futures.items():
                if future.done():
                    if future.exception():
                        counts["failed"] += 1
                    else:
                        counts["completed"] += 1
                elif future.running():
                    counts["running"] += 1
                elif future.cancelled():
                    counts["cancelled"] += 1
                else:
                    counts["pending"] += 1
        
        return counts


# Singleton instance to be used across the application
thread_manager = ThreadManager()
