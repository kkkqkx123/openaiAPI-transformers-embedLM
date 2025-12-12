# Embeddings endpoint test
POST http://localhost:9000/v1/embeddings
Content-Type: application/json

{
    "input": "Hello world",
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "encoding_format": "float"
}
HTTP 200
[Asserts]
jsonpath "$.object" == "list"
jsonpath "$.data" count == 1
jsonpath "$.data[0].object" == "embedding"
jsonpath "$.data[0].embedding" count == 384
jsonpath "$.model" == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
jsonpath "$.usage.prompt_tokens" > 0