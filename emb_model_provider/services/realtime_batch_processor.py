"""
Real-time batch processing module for embedding model provider.

This module provides real-time batch collection with timeout mechanism
to avoid long hangs when processing small batches.
"""

import asyncio
import time
from typing import List, Optional, Any
from dataclasses import dataclass
from emb_model_provider.core.logging import get_logger
from emb_model_provider.core.config import Config
from emb_model_provider.api.embeddings import EmbeddingRequest, EmbeddingData


logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Represents a single request waiting to be batched"""
    request: EmbeddingRequest
    future: asyncio.Future
    timestamp: float


class RealtimeBatchProcessor:
    """
    Real-time batch processor with timeout mechanism to avoid long hangs.
    
    Collects incoming requests and processes them in batches. If a sufficient
    number of requests are available, processes immediately. Otherwise, waits
    for a timeout period before processing with whatever requests are available.
    The implementation uses event-driven scheduling to optimize performance.
    """
    
    def __init__(self, config: Config, embedding_service: Any) -> None:
        self.config = config
        self.embedding_service = embedding_service
        self.max_batch_size = config.max_batch_size
        self.min_batch_size = config.min_batch_size
        self.max_wait_time = config.max_wait_time_ms / 1000.0  # Convert to seconds
        self.hard_timeout = config.hard_timeout_additional_seconds  # Extra timeout after max_wait_time to ensure processing
        
        # Internal request queue
        self._requests: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        
        # Event to signal when new requests arrive
        self._requests_available_event = asyncio.Event()
        
        # Background task for processing
        self._background_task: Optional[asyncio.Task[None]] = None
        
        # Maximum queue size to prevent memory leaks
        self._max_queue_size = config.max_batch_size * 10  # Allow some backlog but prevent unlimited growth
        
        logger.info(
            f"RealtimeBatchProcessor initialized: "
            f"max_batch_size={self.max_batch_size}, "
            f"min_batch_size={self.min_batch_size}, "
            f"max_wait_time={self.max_wait_time}s, "
            f"hard_timeout={self.hard_timeout}s"
        )
    
    async def submit_request(self, request: EmbeddingRequest) -> List[EmbeddingData]:
        """
        Submit a request for batch processing.
        
        Args:
            request: The embedding request to process
            
        Returns:
            List of embedding results
        """
        future: asyncio.Future[List[EmbeddingData]] = asyncio.Future()
        
        # Create a batch request
        batch_request = BatchRequest(
            request=request,
            future=future,
            timestamp=time.time()
        )
        
        # Add to the internal queue
        async with self._lock:
            # Check queue size to prevent memory leaks
            if len(self._requests) >= self._max_queue_size:
                future.set_exception(Exception("Request queue is full, please try again later"))
                return await future
            
            self._requests.append(batch_request)
            current_size = len(self._requests)
        
        # Check if we should process immediately (outside of lock to avoid deadlock)
        should_process = (
            current_size >= self.max_batch_size or
            (current_size >= self.min_batch_size and self._has_request_waited_too_long())
        )
        
        logger.debug(f"Request submitted, queue size: {current_size}, should_process_now: {should_process}")
        
        # Signal that requests are available
        self._requests_available_event.set()
        
        # Process immediately if conditions are met
        if should_process:
            # Use create_task to avoid blocking the submit_request call
            asyncio.create_task(self._process_batch())
        
        try:
            return await future
        except Exception:
            # If the future is cancelled or fails, ensure it's removed from the queue
            async with self._lock:
                if batch_request in self._requests:
                    self._requests.remove(batch_request)
            raise
    
    def _has_request_waited_too_long(self) -> bool:
        """
        Check if any request has waited longer than max_wait_time.
        This method is called outside of lock to avoid deadlock.
        
        Returns:
            True if any request has waited too long
        """
        # Get a snapshot of requests to check timestamps
        # This is a best-effort check without locking
        if not self._requests:
            return False
            
        oldest_timestamp = min(req.timestamp for req in self._requests)
        wait_time = time.time() - oldest_timestamp
        return wait_time >= self.max_wait_time
    
    async def _process_batch(self) -> None:
        """
        Process the current batch of requests.
        """
        # Get pending requests
        requests_to_process = None
        async with self._lock:
            if self._requests:
                requests_to_process = self._requests[:]
                self._requests.clear()
        
        if not requests_to_process:
            return
        
        logger.info(f"Processing batch of {len(requests_to_process)} requests")
        
        # Extract inputs from all requests
        # Note: This implementation assumes all requests use the same model
        all_inputs = []
        for batch_req in requests_to_process:
            inputs = batch_req.request.input if isinstance(batch_req.request.input, list) else [batch_req.request.input]
            all_inputs.extend(inputs)
        
        try:
            # Process all inputs together using the embedding service
            embedding_data_list: List[EmbeddingData] = self.embedding_service.generate_embeddings(all_inputs)
            
            # Distribute results back to individual requests
            idx = 0
            for batch_req in requests_to_process:
                if batch_req.future.done():
                    # Skip if future is already cancelled or done
                    idx += 1 if not isinstance(batch_req.request.input, list) else len(batch_req.request.input)
                    continue
                    
                inputs_count = 1 if not isinstance(batch_req.request.input, list) else len(batch_req.request.input)
                request_results = embedding_data_list[idx:idx + inputs_count]
                
                # Update result indices to match this specific request
                for i, data in enumerate(request_results):
                    data.index = i
                
                batch_req.future.set_result(request_results)
                idx += inputs_count
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all requests in the batch that aren't already done
            for batch_req in requests_to_process:
                if not batch_req.future.done():
                    batch_req.future.set_exception(e)
    
    async def _process_loop(self) -> None:
        """
        Background processing loop that handles timeout-based batch processing.
        Uses event-driven approach to optimize performance.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for either new requests or the timeout, whichever comes first
                # Calculate next timeout based on existing requests
                sleep_time = 0.1  # Default sleep time if no requests are pending (increased from 0.01)
                
                async with self._lock:
                    if self._requests:
                        # Calculate the time when the oldest request will reach max_wait_time
                        oldest_timestamp = min(req.timestamp for req in self._requests)
                        time_to_max_wait = self.max_wait_time - (time.time() - oldest_timestamp)
                        
                        # Calculate the time when the oldest request will reach hard_timeout
                        time_to_hard_timeout = (self.max_wait_time + self.hard_timeout) - (time.time() - oldest_timestamp)
                        
                        # Use the appropriate timeout based on conditions
                        if time_to_max_wait > 0:
                            sleep_time = min(time_to_max_wait, 0.1)  # Don't sleep too long
                        elif time_to_hard_timeout > 0:
                            sleep_time = min(time_to_hard_timeout, 0.1)  # Don't sleep too long
                        else:
                            sleep_time = 0.01  # Minimal sleep when both timeouts are exceeded
                        
                        # Ensure sleep_time is positive
                        sleep_time = max(0.01, sleep_time)
                
                # Wait for either new requests or timeout
                try:
                    await asyncio.wait_for(
                        self._requests_available_event.wait(),
                        timeout=sleep_time
                    )
                    # New requests arrived, clear the event and continue
                    self._requests_available_event.clear()
                except asyncio.TimeoutError:
                    # Timeout reached, check if we need to process due to timeout
                    pass
                
                # Check if we should process based on timeout
                should_process = False
                async with self._lock:
                    current_size = len(self._requests)
                    
                    if current_size > 0 and self._requests:
                        oldest_timestamp = min(req.timestamp for req in self._requests)
                        wait_time = time.time() - oldest_timestamp
                        
                        # If the oldest request has waited beyond max_wait_time and we have at least min_batch_size,
                        # or if we have reached max_batch_size, process immediately
                        should_process_regular = (
                            (wait_time >= self.max_wait_time and current_size >= self.min_batch_size) or
                            current_size >= self.max_batch_size
                        )
                        
                        # Check for hard timeout - this ensures ANY requests are processed after hard_timeout
                        should_process_hard_timeout = wait_time >= (self.max_wait_time + self.hard_timeout)
                        
                        should_process = should_process_regular or should_process_hard_timeout
                        
                if should_process:
                    await self._process_batch()
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)  # Longer sleep on error
    
    async def start(self) -> None:
        """Start the background processing loop."""
        self._background_task = asyncio.create_task(self._process_loop())
        logger.info("RealtimeBatchProcessor started")
    
    async def stop(self) -> None:
        """Stop the background processing loop."""
        if self._background_task:
            self._stop_event.set()
            # Wake up the processing loop if it's waiting
            self._requests_available_event.set()
            
            # Cancel any pending futures
            async with self._lock:
                for batch_req in self._requests:
                    if not batch_req.future.done():
                        batch_req.future.cancel()
                self._requests.clear()
            
            # Wait for the background task to finish with timeout
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Background task did not finish within timeout, cancelling")
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("RealtimeBatchProcessor stopped")
    
    async def get_queue_size(self) -> int:
        """Get the current number of pending requests."""
        async with self._lock:
            return len(self._requests)