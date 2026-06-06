"""Default configuration values for FastCode runtime."""

from typing import Any


def get_default_config() -> dict[str, Any]:
    """Get default configuration.

    Categories:
      [ESSENTIAL] — deployment-specific; users change per environment
      [TUNABLE]  — power-user knobs with sensible defaults
      [INTERNAL] — algorithm internals; rarely need adjustment

    Total: 54 parameters across 13 sections.
    Config file (config/config.yaml) may contain additional sections
    (query, graph, agent, docs_integration) with their own defaults
    defined inline at consumption sites.
    """
    return {
        # ── storage ──────────────────────────────────────────────
        "storage": {
            "backend": "sqlite",  # [ESSENTIAL] "sqlite" or "postgres"
            "postgres_dsn": "",  # [ESSENTIAL] PostgreSQL connection string
            "pool_min": 1,  # [INTERNAL] connection pool minimum
            "pool_max": 8,  # [TUNABLE]  connection pool maximum
        },
        # ── repository ───────────────────────────────────────────
        "repository": {
            "clone_depth": 1,  # [INTERNAL] git shallow clone depth
            "max_file_size_mb": 5,  # [TUNABLE]  skip files larger than this
            "backup_directory": "./repo_backup",  # [TUNABLE] backup location
            "exclude_site_packages": False,  # [TUNABLE] ignore vendored site-packages
            "ignore_patterns": [
                "*.pyc",
                "__pycache__",
                "node_modules",
                ".git",
            ],  # [TUNABLE]
            "supported_extensions": [  # [TUNABLE]  file extensions to index
                ".py",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".java",
                ".go",
                ".rs",
                ".cs",
                ".c",
                ".h",
                ".cc",
                ".cpp",
                ".cxx",
                ".hh",
                ".hpp",
                ".hxx",
                ".zig",
                ".f",
                ".for",
                ".f77",
                ".f90",
                ".f95",
                ".f03",
                ".f08",
                ".jl",
            ],
        },
        # ── parser ───────────────────────────────────────────────
        "parser": {
            "extract_docstrings": True,  # [INTERNAL]
            "extract_comments": True,  # [INTERNAL]
            "extract_imports": True,  # [INTERNAL]
        },
        # ── embedding ────────────────────────────────────────────
        "embedding": {
            "provider": "ollama",  # [ESSENTIAL] "ollama" or "sentence_transformers"
            "model": "bge-large-en-v1.5",  # [ESSENTIAL] embedding model name
            "ollama_url": "http://127.0.0.1:11434/api/embeddings",  # [ESSENTIAL]
            "device": "cpu",  # [TUNABLE]  "auto", "cuda", "mps", "cpu"
            "batch_size": 32,  # [INTERNAL]
        },
        # ── indexing ─────────────────────────────────────────────
        "indexing": {
            "levels": ["file", "class", "function", "documentation"],  # [INTERNAL]
            "allow_direct_index": False,  # [INTERNAL]
        },
        # ── vector_store ─────────────────────────────────────────
        "vector_store": {
            "persist_directory": "./data/vector_store",  # [TUNABLE]
            "distance_metric": "cosine",  # [INTERNAL] similarity metric
            "shard_storage": "compressed",  # [TUNABLE] "compressed" or "npy"
        },
        # ── retrieval ────────────────────────────────────────────
        "retrieval": {
            "semantic_weight": 0.6,  # [TUNABLE]  hybrid search semantic weight
            "keyword_weight": 0.3,  # [TUNABLE]  hybrid search keyword weight
            "graph_weight": 0.1,  # [TUNABLE]  hybrid search graph weight
            "max_results": 5,  # [TUNABLE]  max results per query
            "retrieval_backend": "pg_hybrid",  # [ESSENTIAL] "pg_hybrid" or "local"
            "graph_expansion_backend": "ir",  # [ESSENTIAL] "ir" or "graph_builder"
            "allow_graph_builder_fallback": True,  # [INTERNAL]
        },
        # ── generation ───────────────────────────────────────────
        "generation": {
            "provider": "openai",  # [ESSENTIAL] "openai", "anthropic", or "local"
            "model": "gpt-4-turbo-preview",  # [ESSENTIAL] LLM model name
            "temperature": 0.1,  # [TUNABLE]
            "max_tokens": 2000,  # [TUNABLE]
        },
        # ── evaluation ───────────────────────────────────────────
        "evaluation": {
            "enabled": False,  # [TUNABLE]  enable benchmark/eval mode
            "in_memory_index": False,  # [INTERNAL] keep index in RAM only
            "disable_cache": False,  # [INTERNAL] skip query/embedding cache
            "disable_persistence": False,  # [INTERNAL] skip writing artifacts
            "force_reindex": False,  # [INTERNAL] always rebuild index
        },
        # ── cache ────────────────────────────────────────────────
        "cache": {
            "enabled": True,  # [TUNABLE]
            "backend": "disk",  # [ESSENTIAL] "disk" or "redis"
            "cache_directory": "./data/cache",  # [TUNABLE]
            "cache_queries": False,  # [INTERNAL]
            "redis_host": "localhost",  # [ESSENTIAL] Redis cache host
            "redis_port": 6379,  # [ESSENTIAL] Redis cache port
        },
        # ── logging ──────────────────────────────────────────────
        "logging": {
            "level": "INFO",  # [TUNABLE]  DEBUG, INFO, WARNING, ERROR
            "console": True,  # [INTERNAL]
        },
        # ── terminus ─────────────────────────────────────────────
        "terminus": {
            "endpoint": "",  # [ESSENTIAL] TerminusDB publish endpoint
            "api_key": "",  # [ESSENTIAL] TerminusDB API key
            "timeout_seconds": 15,  # [INTERNAL]
        },
        # ── projection ───────────────────────────────────────────
        "projection": {
            "postgres_dsn": "",  # [ESSENTIAL] projection store DSN
            "enable_leiden": True,  # [TUNABLE]  enable Leiden clustering
            "llm_enabled": True,  # [TUNABLE]  enable LLM label generation
            "llm_timeout_seconds": 8,  # [INTERNAL]
            "llm_max_tokens": 180,  # [INTERNAL]
            "llm_temperature": 0.2,  # [INTERNAL]
            "max_entity_hops": 2,  # [INTERNAL]
            "max_query_hops": 2,  # [INTERNAL]
            "max_chunk_count": 64,  # [INTERNAL],
        },
    }
