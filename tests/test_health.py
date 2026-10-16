import importlib
import pytest

CASES = [
    ("drowsiness_api", "/health", 8003),
    ("object_detection_api", "/health", 8001),
    ("traffic_light_api", "/health", 8002),
    ("traffic_sign_api", "/health", 8004),
    ("api", "/health", 8000),
]


@pytest.mark.parametrize("module_name", ["object_detection_api", "traffic_light_api"])
def test_yolo_health_unavailable(module_name):
    mod = importlib.import_module(module_name)
    app = mod.app
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body or "classes" in body


def test_drowsiness_health_unavailable():
    import importlib
    mod = importlib.import_module("drowsiness_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "unavailable" in r.json().get("status", "unavailable") or r.json().get("status") in ("unavailable", "healthy")


def test_traffic_sign_health_unavailable():
    import importlib
    mod = importlib.import_module("traffic_sign_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") in ("unavailable", "healthy")


def test_driver_health_unavailable():
    import importlib
    mod = importlib.import_module("api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") in ("unavailable", "healthy")


def test_object_detection_invalid_conf():
    import importlib
    mod = importlib.import_module("object_detection_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = client.post("/detect_base64", json={"image": b64, "conf": 2.0})
        assert r.status_code == 400


def test_traffic_light_invalid_conf():
    import importlib
    mod = importlib.import_module("traffic_light_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = client.post("/detect_traffic_light_base64", json={"image": b64, "conf": 2.0})
        assert r.status_code == 400


def test_drowsiness_no_face_returns_unknown_or_unavailable():
    import importlib
    mod = importlib.import_module("drowsiness_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = client.post("/detect_drowsiness_base64", json={"image": b64})
        assert r.status_code == 200
        assert "state" in r.json()


def test_traffic_sign_decode_error():
    import importlib
    mod = importlib.import_module("traffic_sign_api")
    from starlette.testclient import TestClient
    with TestClient(mod.app) as client:
        r = client.post("/classify_base64", json={"image": "not-base64!!!"})
        assert r.status_code == 400
