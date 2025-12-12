# List models endpoint test
GET http://localhost:9000/v1/models
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.data" count >= 1
jsonpath "$.data[0].id" matches /\w+/