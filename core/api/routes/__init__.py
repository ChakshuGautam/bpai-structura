from core.api.routes.document_types import router as document_types_router
from core.api.routes.convert import router_v1 as convert_router_v1, router_v2 as convert_router_v2, router_v3 as convert_router_v3

# Export all routers
__all__ = ["document_types_router", "convert_router_v1", "convert_router_v2", "convert_router_v3"]
