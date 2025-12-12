# Root endpoint test
GET http://localhost:9000/
HTTP 200
[Asserts]
jsonpath "$.message" == "Embedding Model Provider API"
jsonpath "$.endpoints" count == 3