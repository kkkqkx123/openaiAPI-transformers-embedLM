"""
Tests for the real-time batch processor timeout mechanism.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock
from emb_model_provider.services.realtime_batch_processor import RealtimeBatchProcessor
from emb_model_provider.core.config import Config
from emb_model_provider.api.embeddings import EmbeddingRequest, EmbeddingData


def create_mock_config():
    """Create a mock configuration for testing."""
    config = Config()
    config.max_batch_size = 4
    config.min_batch_size = 2
    config.max_wait_time_ms = 100  # 100ms
    return config


def create_mock_embedding_service():
    """Create a mock embedding service for testing."""
    service = Mock()
    service.generate_embeddings = Mock(return_value=[
        EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
        EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        EmbeddingData(embedding=[0.7, 0.8, 0.9], index=2)
    ])
    return service


@pytest.mark.asyncio
async def test_realtime_batch_processor_with_min_batch_size():
    """Test that batches are processed when reaching min batch size within max wait time."""
    config = create_mock_config()
    service = create_mock_embedding_service()
    processor = RealtimeBatchProcessor(config, service)
    
    # Start the processor
    await processor.start()
    
    try:
        # Create 2 requests (reaching min_batch_size)
        request1 = EmbeddingRequest(input="Hello", model=config.model_name)
        request2 = EmbeddingRequest(input="World", model=config.model_name)
        
        # Submit the requests
        task1 = asyncio.create_task(processor.submit_request(request1))
        task2 = asyncio.create_task(processor.submit_request(request2))
        
        # Wait for results
        result1 = await task1
        result2 = await task2
        
        # Check that the embedding service was called
        service.generate_embeddings.assert_called_once()
        args, _ = service.generate_embeddings.call_args
        assert len(args[0]) == 2  # Two inputs processed together
        
    finally:
        await processor.stop()


@pytest.mark.asyncio
async def test_realtime_batch_processor_with_max_batch_size():
    """Test that batch is processed immediately when reaching max batch size."""
    config = create_mock_config()
    service = create_mock_embedding_service()
    processor = RealtimeBatchProcessor(config, service)
    
    # Start the processor
    await processor.start()
    
    try:
        # Create 4 requests (reaching max_batch_size)
        requests = [
            EmbeddingRequest(input=f"Text {i}", model=config.model_name)
            for i in range(4)
        ]
        
        # Submit the requests
        tasks = [asyncio.create_task(processor.submit_request(req)) for req in requests]
        
        # Wait for results
        results = await asyncio.gather(*tasks)
        
        # Check that the embedding service was called
        service.generate_embeddings.assert_called_once()
        args, _ = service.generate_embeddings.call_args
        assert len(args[0]) == 4  # Four inputs processed together
        
    finally:
        await processor.stop()


@pytest.mark.asyncio
async def test_realtime_batch_processor_timeout():
    """Test that batch is processed with adaptive timeout when less than min_batch_size."""
    config = create_mock_config()
    service = create_mock_embedding_service()
    processor = RealtimeBatchProcessor(config, service)
    
    # Start the processor
    await processor.start()
    
    try:
        # Create 1 request (less than min_batch_size of 2)
        request = EmbeddingRequest(input="Hello", model=config.model_name)
        
        start_time = time.time()
        result = await processor.submit_request(request)
        end_time = time.time()
        
        # The request should be processed with adaptive timeout (much faster than hard timeout)
        processing_time = end_time - start_time
        
        # With adaptive timeout, single requests should be processed much faster than hard timeout
        # The optimization reduces single request latency from ~1.1s to ~0.1s
        min_expected = 0.05  # At least 50ms for processing
        max_expected = (config.max_wait_time_ms / 1000.0) + 0.05  # max_wait_time + small buffer
        
        assert processing_time >= min_expected
        assert processing_time <= max_expected
        
        # Verify the embedding service was called with the single input
        service.generate_embeddings.assert_called_once()
        args, _ = service.generate_embeddings.call_args
        assert len(args[0]) == 1  # One input processed
        
    finally:
        await processor.stop()


@pytest.mark.asyncio
async def test_realtime_batch_processor_hard_timeout():
    """Test the hard timeout mechanism that forces processing of small batches."""
    config = create_mock_config()
    # Set a very short max_wait_time and min_batch_size > 1 to trigger hard timeout
    config.max_wait_time_ms = 50  # 50ms
    config.min_batch_size = 4  # Require 4 requests normally
    config.max_batch_size = 8
    
    service = create_mock_embedding_service()
    processor = RealtimeBatchProcessor(config, service)
    
    # Start the processor
    await processor.start()
    
    try:
        # Create 1 request (much less than min_batch_size)
        request = EmbeddingRequest(input="Hello", model=config.model_name)
        
        start_time = time.time()
        result = await processor.submit_request(request)
        end_time = time.time()
        
        # The request should be processed after max_wait_time + hard_timeout (1000ms + 50ms)
        processing_time = end_time - start_time
        
        # Should take at least max_wait_time + some but not more than max_wait_time + hard_timeout
        min_expected = config.max_wait_time_ms / 1000.0
        max_expected = (config.max_wait_time_ms / 1000.0) + processor.hard_timeout + 0.5  # Add buffer for processing
        
        assert processing_time >= min_expected
        assert processing_time <= max_expected
        
        # Verify the embedding service was called with the single input
        service.generate_embeddings.assert_called_once()
        args, _ = service.generate_embeddings.call_args
        assert len(args[0]) == 1  # One input processed
        
    finally:
        await processor.stop()


@pytest.mark.asyncio
async def test_multiple_requests_with_hard_timeout():
    """Test processing of multiple requests that are less than min batch size but exceed hard timeout."""
    config = create_mock_config()
    # Set min batch size higher to ensure timeout behavior
    config.min_batch_size = 5
    config.max_wait_time_ms = 50  # 50ms max wait
    
    service = create_mock_embedding_service()
    processor = RealtimeBatchProcessor(config, service)
    
    # Start the processor
    await processor.start()
    
    try:
        # Create 2 requests (less than min_batch_size of 5)
        requests = [
            EmbeddingRequest(input="Hello", model=config.model_name),
            EmbeddingRequest(input="World", model=config.model_name)
        ]
        
        start_time = time.time()
        
        # Submit the requests
        tasks = [asyncio.create_task(processor.submit_request(req)) for req in requests]
        
        # Wait for results
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should be processed due to hard timeout after max_wait_time + hard_timeout
        min_expected = config.max_wait_time_ms / 1000.0
        max_expected = (config.max_wait_time_ms / 1000.0) + processor.hard_timeout + 0.5  # Add buffer
        
        assert processing_time >= min_expected
        assert processing_time <= max_expected
        
        # Verify the embedding service was called with the two inputs
        service.generate_embeddings.assert_called_once()
        args, _ = service.generate_embeddings.call_args
        assert len(args[0]) == 2  # Two inputs processed together
        
    finally:
        await processor.stop()