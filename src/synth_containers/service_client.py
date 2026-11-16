"""Client for named Synth operation endpoints, with immutable origin binding."""

from urllib.parse import quote

import httpx

from .package_models import content_digest


class ServiceOperationClient:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self.client, self.base_url = client, base_url.rstrip("/")

    async def metadata(self):
        response = await self.client.get(self.base_url + "/.well-known/synth-service")
        response.raise_for_status()
        value = response.json()
        if value.get("operation_contract") != "synth.service-operations.v1":
            raise ValueError("service_operation_contract_unsupported")
        return value

    async def submit(self, operation_id: str, payload: dict, *, origin_instance: str):
        response = await self.client.post(
            self.base_url + "/operations",
            json={
                "operation_id": operation_id,
                "input": payload,
                "expected_origin_instance": origin_instance,
            },
        )
        response.raise_for_status()
        return self.verify(
            response.json(),
            operation_id=operation_id,
            origin_instance=origin_instance,
            input_sha256=content_digest(payload)[7:],
        )

    @staticmethod
    def verify(receipt, *, operation_id, origin_instance, input_sha256):
        if (
            receipt.get("origin_instance") != origin_instance
            or receipt.get("input_sha256") != input_sha256
            or receipt.get("record", {}).get("operation_id") != operation_id
        ):
            raise ValueError("service_operation_receipt_mismatch")
        return receipt

    async def read(
        self, operation_id: str, *, origin_instance: str, input_sha256: str, cancel=False
    ):
        response = await self.client.request(
            "POST" if cancel else "GET",
            self.base_url
            + "/operations/"
            + quote(operation_id, safe="")
            + ("/cancel" if cancel else ""),
            params={"origin_instance": origin_instance, "input_sha256": input_sha256},
        )
        response.raise_for_status()
        return self.verify(
            response.json(),
            operation_id=operation_id,
            origin_instance=origin_instance,
            input_sha256=input_sha256,
        )
