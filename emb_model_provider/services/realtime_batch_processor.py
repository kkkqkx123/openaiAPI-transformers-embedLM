"""
Real-time batch processing module for embedding model provider.

This module provides real-time batch collection with timeout mechanism
to avoid long hangs when processing small batches.
"""

import asyncio
import time
from typing import List, Dict, Callable, Any, Optional, Tuple
from dataclasses import dataclass
from threading import Thread, Lock
import asyncio
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


@dataclass
class ProcessedBatchResult:
    """Result from processing a batch"""
    results: List[EmbeddingData]
    start_time: float
    end_time: float


class RealtimeBatchProcessor:
    """
    Real-time batch processor with timeout mechanism to avoid long hangs.
    
    Collects incoming requests and processes them in batches. If a sufficient
    number of requests are available, processes immediately. Otherwise, waits
    for a timeout period before processing with whatever requests are available.
    """
    
    def __init__(self, config: Config, embedding_service):
        self.config = config
        self.embedding_service = embedding_service
        self.max_batch_size = config.max_batch_size
        self.min_batch_size = config.min_batch_size
        self.max_wait_time = config.max_wait_time_ms / 1000.0  # Convert to seconds
        self.hard_timeout = config.hard_timeout_additional_seconds  # Extra timeout after max_wait_time to ensure processing
        
        # Internal request queue
        self._requests: List[BatchRequest] = []
        self._lock = Lock()
        self._stop_event = asyncio.Event()
        
        # Background task for processing
        self._background_task = None
        
        logger.info(
            f"RealtimeBatchProcessor initialized: "
            f"max_batch_size={self.max_batch_size}, "
            f"min_batch_size={self.min_batch_size}, "
            f"max_wait_time={self.max_wait_time}s"
        )
    
    async def submit_request(self, request: EmbeddingRequest) -> List[EmbeddingData]:
        """
        Submit a request for batch processing.
        
        Args:
            request: The embedding request to process
            
        Returns:
            List of embedding results
        """
        future = asyncio.Future()
        
        # Create a batch request
        batch_request = BatchRequest(
            request=request,
            future=future,
            timestamp=time.time()
        )
        
        # Add to the internal queue
        with self._lock:
            self._requests.append(batch_request)
            should_process = self._should_process_now()
        
        logger.debug(f"Request submitted, queue size: {len(self._requests)}, should_process_now: {should_process}")
        
        # Process immediately if conditions are met, otherwise wait
        if should_process:
            await self._process_batch()
        
        return await future
    
    def _should_process_now(self) -> bool:
        """
        Determine if we should process the batch immediately.
        
        Returns:
            True if batch should be processed immediately
        """
        with self._lock:
            current_size = len(self._requests)
        
        # Process immediately if we've reached max batch size
        if current_size >= self.max_batch_size:
            return True
            
        # Process immediately if we've reached min batch size and max wait time is exceeded
        # (for requests that have been waiting)
        with self._lock:
            if current_size >= self.min_batch_size and self._requests:
                oldest_timestamp = min(req.timestamp for req in self._requests)
                wait_time = time.time() - oldest_timestamp
                if wait_time >= self.max_wait_time:
                    return True
        
        return False
    
    async def _process_batch(self):
        """
        Process the current batch of requests.
        """
        # Get pending requests
        requests_to_process = None
        with self._lock:
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
            embedding_data_list = self.embedding_service.generate_embeddings(all_inputs)
            
            # Distribute results back to individual requests
            idx = 0
            for batch_req in requests_to_process:
                inputs_count = 1 if not isinstance(batch_req.request.input, list) else len(batch_req.request.input)
                request_results = embedding_data_list[idx:idx + inputs_count]
                
                # Update result indices to match this specific request
                for i, data in enumerate(request_results):
                    data.index = i
                
                batch_req.future.set_result(request_results)
                idx += inputs_count
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all requests in the batch
            for batch_req in requests_to_process:
                batch_req.future.set_exception(e)
    
    async def _process_loop(self):
        """
        Background processing loop that handles timeout-based batch processing.
        """
        while not self._stop_event.is_set():
            try:
                # Check if we should process based on timeout
                with self._lock:
                    current_size = len(self._requests)
                
                if current_size > 0:
                    # Check oldest request timestamp
                    with self._lock:
                        if self._requests:
                            oldest_timestamp = min(req.timestamp for req in self._requests)
                            wait_time = time.time() - oldest_timestamp
                            
                            # If the oldest request has waited beyond max_wait_time and we have at least 1 request,
                            # or if we have enough requests for min batch size
                            should_process_regular = (
                                (wait_time >= self.max_wait_time and current_size >= self.min_batch_size) or
                                current_size >= self.max_batch_size
                            )
                            
                            # Check for hard timeout - this ensures ANY requests are processed after hard_timeout
                            # to prevent long hangs for small batches
                            should_process_hard_timeout = wait_time >= (self.max_wait_time + self.hard_timeout)
                            
                            should_process = should_process_regular or should_process_hard_timeout
                            
                    if should_process:
                        await self._process_batch()
                    # No else clause needed - if not processing, continue waiting
                
                # Wait a bit before checking again
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)  # Longer sleep on error
    
    async def start(self):
        """Start the background processing loop."""
        self._background_task = asyncio.create_task(self._process_loop())
        logger.info("RealtimeBatchProcessor started")
    
    async def stop(self):
        """Stop the background processing loop."""
        if self._background_task:
            self._stop_event.set()
            await self._background_task
            logger.info("RealtimeBatchProcessor stopped")
    
    def get_queue_size(self) -> int:
        """Get the current number of pending requests."""
        with self._lock:
            return len(self._requests)