# Data sources, attribution, and reproducibility

This document separates research-code licensing from input-data rights. The MIT software license does not relicense any external dataset. Paths below are relative to the research project directory containing configs/, data/, and experiments/.

## Dataset register

| Dataset | Study use and expected paths | Source and rights |
|---|---|---|
| BTS Reporting Carrier On-Time Performance | January–June 2024 monthly ZIPs under data/original/; cleaned flight data under data/clean/. | U.S. Bureau of Transportation Statistics, [TranStats](https://www.transtats.bts.gov/), [on-time download interface](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr). The project records this federal source as public domain; a separate dataset-specific license declaration was not retrieved in this audit. BTS is the source, not an endorser. |
| Online Retail II | Original online_retail_II.xlsx under data/original/; preprocessed retail data under data/clean/. | Daqing Chen, [UCI dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii), [dataset DOI](https://doi.org/10.24432/C5CG6D). The UCI record explicitly specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| Chicago Food Inspections | Food_Inspections.csv under data/original/; cleaned controlled-experiment data and raw monthly chronological replay. | City of Chicago, Department of Public Health, [dataset 4ijn-s7e5](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5). Official metadata specifies “See Terms of Use”; [City Data Terms](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) apply. |
| Chicago city boundary | Chicago_City_Boundary.csv under data/original/; coordinate bounding box and polygon checks. | City of Chicago, [dataset qqq8-j68g](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-City/qqq8-j68g). Official metadata specifies “See Terms of Use”; the same [City Data Terms](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) apply. |
| Tax and Beers | Aligned dirty.csv and clean.csv files under data/external/tax/ and data/external/beers/. | [Raha/Baran benchmark collection](https://github.com/BigDaMa/raha/tree/master/datasets). The repository's [Apache 2.0 license](https://github.com/BigDaMa/raha/blob/master/LICENSE.md) is preserved separately. A distinct license for each underlying table was not independently verified. |

## Attribution and transformations

Online Retail II attribution: Daqing Chen, Online Retail II, UCI Machine Learning Repository, DOI 10.24432/C5CG6D, used under CC BY 4.0. This research creates filtered/cleaned subsets, shuffled batches, statistical profiles, and controlled drift/error variants; these transformations are not the original UCI dataset.

Chicago attribution: City of Chicago Data Portal, Food Inspections and Boundaries—City. This research creates cleaned subsets, monthly replay summaries, coordinate checks, and controlled experimental variants. Include the full City derivative-application notice in THIRD_PARTY_NOTICES.md wherever the application or a Chicago-derived archive is offered for download.

BTS attribution: U.S. Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, January–June 2024. The research filters specified records and creates experimental batches and injected variants. Derived research artifacts are not official BTS products.

Raha attribution: Mohammad Mahdavi and collaborators, Raha: A Configuration-Free Error Detection System, SIGMOD 2019; and the Raha/Baran benchmark maintainers. The project uses the supplied dirty/clean pairs as evaluation data, not the Raha system as a comparator. Shuffled batch replay and numeric-equivalence label normalization are research-specific transformations.

## Snapshot identity

The supplied data/original/download_manifest.json records a download on 5 October 2026, including six monthly BTS URLs, the UCI workbook source, and the Chicago inspection export. The boundary source is additionally identified in ingestion/geo.py. Run manifests record input hashes and cleaning/replay settings. Preserve the release's input checksums when reproducing the exact experiment.

Chicago's endpoints are live: downloading them again can yield more records and a different chronological replay. A current download is not silently substituted for the frozen snapshot. Likewise, this release does not assign immutable version identities to third-party datasets merely because their filenames are unchanged.

## Download and restore

The existing scripts/download_data.py and scripts/download_benchmarks.py fetch the declared sources. Downloaders use the configured names and expected locations; the documented environment and preparation commands then build cleaned inputs and batches. Rebuilding from a new live snapshot is a fresh-data run, not exact verification of the saved outputs.

For exact reproduction, use the release's data companion asset if one is distributed and compare its file hashes with the release manifest. Keep its provider notices with the data. If data files are not bundled, the code and saved-result review remain usable, while a full replay additionally requires restoring the source inputs.

## Packaging in this release

Original and cleaned data are distributed in the separate research-assets archive, outside ordinary Git history. The archive contains the supplied fixed snapshot, the data-source register, provider notices and file hashes. These files retain provider terms; they are not newly licensed under MIT.

Tax/Beers files are preserved as supplied from the Raha benchmark repository, accompanied by its Apache 2.0 declaration. Separate rights statements for the underlying source tables were not independently resolved. This provenance limitation is retained rather than represented as a new permission grant. The original fetch script remains available.

## Verification status

- Verified directly: UCI's CC BY 4.0 declaration; Chicago's “See Terms of Use” metadata for both datasets and the canonical City terms; Raha's repository-level Apache 2.0 declaration.
- Verified provenance but not an independent dataset-specific license declaration: BTS monthly input files.
- Not independently resolved: separate source-level rights statements for the underlying Tax and Beers tables. They are not newly labelled MIT.
- License/source metadata was checked on 24 October 2026. Provider terms and endpoints can change.
