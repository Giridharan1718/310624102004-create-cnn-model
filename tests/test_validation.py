import base64
import io
import pytest
from PIL import Image

from validation_utils import parse_conf, decode_b64_image, ALLOWED_IMAGE_MIME, read_upload_bytes
from fastapi import UploadFile
from starlette.datastructures import Headers
from starlette.testclient import TestClient


def test_parse_conf_valid():
    assert parse_conf("0.5") == 0.5


def test_parse_conf_invalid():
    with pytest.raises(Exception):
        parse_conf("2.0")
    with pytest.raises(Exception):
        parse_conf("abc")


def test_decode_b64_image():
    img = Image.new("RGB", (4, 4), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    out = decode_b64_image(b64)
    assert out.size == (4, 4)


def test_decode_b64_empty():
    with pytest.raises(Exception):
        decode_b64_image("")


@pytest.mark.asyncio
async def test_read_upload_bytes_empty_upload():
    class Dummy:
        filename = "x.png"
        content_type = "image/png"
        async def read(self, *_):
            return b""
    u = Dummy()
    # Emulate UploadFile-like read
    with pytest.raises(Exception):
        await read_upload_bytes(u)
