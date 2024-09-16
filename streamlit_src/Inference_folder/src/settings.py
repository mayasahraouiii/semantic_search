import os
import json
from types import MappingProxyType

elastic_version = os.getenv("ELASTIC_ORGS_VERSION")
experts_elastic_version = os.getenv("ELASTIC_EXPERTS_VERSION")
funding_elastic_version = os.getenv("ELASTIC_FUNDING_VERSION")


def get_configurations() -> MappingProxyType:
    data = {
    "USERS_DATABASE": "users",
    "PROJECTS_CONTAINER": "demo-projects",
    "NOSTRADAMUS_DATABASE": "nostradamus",
    "SPOTLIGHTS_CONTENT_CONTAINER": "spotlight-codex",
    "INISGHTS_CONTENT_CONTAINER": "insights-codex"
    }

    # the configurations dictionary is immutable
    configurations = MappingProxyType(
        {
            # elastic search configurations
            "ELASTICSEARCH_COMPANY_INDEX": f"search-orgs_{elastic_version}_dev",
            "ELASTICSEARCH_EXPERT_INDEX": f"search-experts_{experts_elastic_version}_dev",
            "ELASTICSEARCH_NEWS_INDEX": f"search-news_v1.0.1_dev",
            "ELASTICSEARCH_FUNDING_ROUND_INDEX": f"search-funding_rounds_{funding_elastic_version}_dev",
            "USERS_DATABASE": data.get("USERS_DATABASE"),
            "PROJECTS_CONTAINER": data.get("PROJECTS_CONTAINER"),
            "NOSTRADAMUS_DATABASE": data.get("NOSTRADAMUS_DATABASE"),
            "SPOTLIGHTS_CONTENT_CONTAINER": data.get(
                "SPOTLIGHTS_CONTENT_CONTAINER"
            ),
            "INSIGHTS_CONTAINER": data.get("INISGHTS_CONTENT_CONTAINER"),
        }
    )

    return configurations
configurations = get_configurations()
