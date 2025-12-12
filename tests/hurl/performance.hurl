# Performance metrics endpoint test
GET http://localhost:9000/v1/performance
HTTP 200
[Asserts]
jsonpath "$.total_requests_processed" >= 0

# Reset performance metrics
POST http://localhost:9000/v1/performance/reset
HTTP 200
[Asserts]
jsonpath "$.message" == "Performance metrics reset successfully"