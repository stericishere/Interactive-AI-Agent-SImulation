"""
Dating Show API URLs
URL configuration for the enhanced dating show API endpoints
"""

from django.urls import path, re_path
from . import views

urlpatterns = [
    # ==================== WEB INTERFACE ROUTES ====================
    
    # Main dashboard
    re_path(r'^$', views.dating_show_home, name='dating_show_home'),
    
    # Agent detail views
    re_path(r'^agent/(?P<agent_id>[\w-]+)/$', views.agent_detail_enhanced, name='agent_detail_enhanced'),
    
    # Governance interface
    re_path(r'^governance/$', views.governance_panel, name='governance_panel'),
    
    # Social network visualization
    re_path(r'^social/$', views.social_network, name='social_network'),
    
    # Analytics dashboard
    re_path(r'^analytics/$', views.analytics_dashboard, name='analytics'),
    
    
    # ==================== REST API ENDPOINTS ====================
    
    # Agent endpoints
    re_path(r'^api/agents/$', views.api_agents_list, name='api_agents_list'),
    re_path(r'^api/agents/(?P<agent_id>[\w-]+)/$', views.api_agent_detail, name='api_agent_detail'),
    re_path(r'^api/agents/(?P<agent_id>[\w-]+)/state/$', views.api_agent_state_update, name='api_agent_state_update'),
    
    # Governance endpoints
    re_path(r'^api/governance/votes/$', views.api_governance_votes, name='api_governance_votes'),
    re_path(r'^api/governance/votes/(?P<vote_id>[\w-]+)/cast/$', views.api_cast_vote, name='api_cast_vote'),
    
    # Simulation control endpoints
    re_path(r'^api/simulation/status/$', views.api_simulation_status, name='api_simulation_status'),
    re_path(r'^api/simulation/control/$', views.api_simulation_control, name='api_simulation_control'),
    
    # Social network endpoints
    re_path(r'^api/social/network/$', views.api_social_network, name='api_social_network'),
    
    # Real-time update endpoints
    re_path(r'^api/broadcast/update/$', views.api_broadcast_update, name='api_broadcast_update'),
]