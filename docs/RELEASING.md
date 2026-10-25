# Publishing this research artifact

## Repository contents

Extract adaptive-dq-research-v0.1.0-github.zip. The archive contains one repository directory. Commit its contents, not the ZIP as a single source file.

The root LICENSE is already present. Preserve it together with THIRD_PARTY_NOTICES.md and third_party/. Do not replace the data/template terms with MIT.

Use GitHub Desktop or Git to publish the extracted directory to an empty repository. Set the remote to the actual repository you create; no account name or repository URL is assumed here.

```bash
git init -b main
git add .
git commit -m "Release research artifact v0.1.0"
# Add your actual GitHub remote, then push main.
```

Run python release_tools/verify_release.py before committing. CI will verify integrity and run the offline test suite after publication. The exact frozen byte hashes depend on the included .gitattributes; retain that file.

## GitHub release

Create a release tagged v0.1.0 and attach:

1. adaptive-dq-research-v0.1.0-github.zip
2. adaptive-dq-research-assets-v0.1.0.zip
3. Adaptive_DQ_Complete_Paper.pdf
4. SHA256SUMS.txt

The large companion is a release asset rather than a regular Git-tracked file. GitHub currently blocks ordinary repository files above 100 MiB and supports individual release assets below 2 GiB. See [large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Suggested description: Research artifact for adaptive and cost-aware data quality monitoring in evolving batch pipelines.

Suggested topics: data-quality, data-validation, distribution-drift, reproducible-research, python.

## Citation and archival record

After publication, add the real repository URL to CITATION.cff and the README. If an archival service assigns a DOI, add that real DOI and update the manuscript's Data and Code Availability statement. Then regenerate integrity hashes and issue a new release version for the metadata change. Do not invent a DOI or label the unpublished manuscript as a published journal article.

An immutable tagged release with its paper, manifests, source and input snapshots is more useful to a reviewer than a moving branch alone.

## Release checklist

- Confirm all five authors and their order against the approved manuscript.
- Run the release verifier and tests.
- Inspect the paper PDF and ensure it corresponds to the tagged LaTeX.
- Preserve the original MIT and third-party notices.
- Check the supplied checksum list after copying/uploading files.
- Keep the frozen pipeline and configs unchanged; describe new experiments under a new protocol version.

## Notice for the release download page

Paste the following provider notice into the release description when the Chicago-derived archive is offered:

> This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago. The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site. The data provided at this site is subject to change at any time. It is understood that the data provided at this site is being used at one's own risk.
