# Health check endpoint test
GET http://localhost:9000/health
HTTP 200
[Asserts]
jsonpath "$.status" == "healthy"