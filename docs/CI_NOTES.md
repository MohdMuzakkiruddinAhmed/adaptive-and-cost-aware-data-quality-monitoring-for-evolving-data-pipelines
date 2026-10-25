# CI design and local validation

The workflow checks released evidence and runs synthetic tests. Dependency installation uses PyPI, but the tests themselves do not require network access, AWS credentials, external datasets, Java, Spark, Docker or live models.

The true TFDV sidecar test is disabled by leaving ADQ_TFDV_PYTHON empty. Other external-tool tests exercise synthetic exchange formats and translation.

Actions are pinned to full commit identifiers. Release identities were verified from official repositories on 24 October 2026:

- [actions/checkout v7.0.1](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1)
- [actions/setup-python v7.1.0](https://github.com/actions/setup-python/commit/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0)

Repository permissions are read-only and checkout credentials are not persisted. No publication or deployment workflow is included.

Local validation uses Python 3.12.10 with portable dependencies, not the study's locked Python 3.14 environment. A first restricted-Windows run found temporary-directory permission failures; redirecting TEMP and TMP to a writable workspace avoids them without changing source. The final outcome is recorded in test_validation.json.
