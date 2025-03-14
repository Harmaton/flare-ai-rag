"""
Configuration for data sources including GitHub repositories, documentation, and blogs.
"""
from datetime import datetime

FLARE_SOURCES: dict[str, list[dict[str, str | dict[str, str] | float]]] = {
    "github_repos": [
        {
            "source_type": "git",
            "path": "https://github.com/flare-foundation/go-flare",
            "metadata": {
                "description": "Main Flare Network repository",
                "category": "core",
            },
            "verification_score": 1.0,
        },
        {
            "source_type": "git",
            "path": "https://dev.flare.network/ftso/overview",
            "metadata": {
                "description": "Flare Documentation",
                "category": "documentation",
            },
            "verification_score": 1.0,
        },
    ],
    "documentation": [
        {
            "source_type": "markdown",
            "path": "https://dev.flare.network/network/getting-started/",
            "metadata": {
                "description": "Official Flare Documentation",
                "category": "documentation",
            },
            "verification_score": 1.0,
        }
    ],
    "blogs": [
        {
            "source_type": "markdown",
            "path": "https://dev.flare.network/fdc/overview",
            "metadata": {
                "description": "Official Flare Blog",
                "category": "blog",
            },
            "verification_score": 0.9,
        }
    ]
}
