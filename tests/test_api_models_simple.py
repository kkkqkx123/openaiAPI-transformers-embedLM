import pytest
import json
from fastapi.testclient import TestClient
from emb_model_provider.main import app


class TestModelsAPISimple:
    """简化的Models API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_models_endpoint_basic(self, client):
        """测试基本的models端点"""
        response = client.get("/v1/models")
        
        # 应该返回200，即使配置有问题也不应该崩溃
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        
        # 打印实际响应用于调试
        print(f"Models response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # 验证响应不为空（应该至少有一个默认模型）
        assert len(data["data"]) > 0, "Models list should not be empty"
        
        # 验证第一个模型的结构
        first_model = data["data"][0]
        assert "id" in first_model
        assert "object" in first_model
        assert "created" in first_model
        assert "owned_by" in first_model
        assert first_model["object"] == "model"
        assert isinstance(first_model["created"], int)
    
    def test_models_endpoint_response_format(self, client):
        """测试响应格式"""
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        
        # 验证OpenAI兼容格式
        assert "object" in data
        assert "data" in data
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
    
    def test_models_endpoint_with_real_config(self, client):
        """使用真实配置测试"""
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # 记录实际返回的模型信息
        print(f"Available models: {[m['id'] for m in data['data']]}")
        
        # 验证所有模型ID都是字符串
        for model in data["data"]:
            assert isinstance(model["id"], str)
            assert len(model["id"]) > 0
    
    def test_models_endpoint_consistency(self, client):
        """测试多次调用的一致性"""
        # 第一次调用
        response1 = client.get("/v1/models")
        data1 = response1.json()
        
        # 第二次调用
        response2 = client.get("/v1/models")
        data2 = response2.json()
        
        # 基本结构应该一致
        assert data1["object"] == data2["object"]
        assert len(data1["data"]) == len(data2["data"])
        
        # 模型列表应该相同（顺序可能不同）
        models1 = {m["id"] for m in data1["data"]}
        models2 = {m["id"] for m in data2["data"]}
        assert models1 == models2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])