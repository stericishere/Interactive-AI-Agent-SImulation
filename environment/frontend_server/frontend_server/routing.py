from django.urls import re_path

from dating_show_api import consumers

websocket_urlpatterns = [
    re_path(r'ws/agent/(?P<agent_id>\w+)/$', consumers.AgentStateConsumer.as_asgi()),
]
