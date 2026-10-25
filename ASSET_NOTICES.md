# Research-assets companion

This archive restores the exact supplied data snapshots and external-validator exchange outputs for artifact version 0.1.0 and protocol v2.2. Extract or restore it into the matching source release using release_tools/restore_assets.py.

The MIT source-code license does not relicense external datasets. Provider terms and provenance follow below. The matching source release includes the preserved provider notices and template sources under third_party/.

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


# Third-party notices and license scope

Verified for this release on 24 October 2026.

The repository's MIT license covers the authors' original research software and other original material to which they hold the necessary rights. It does not replace third-party licenses, data-provider terms, copyright notices, or trademark rights. Cleaning, batching, or injecting errors into a source dataset does not turn that source into MIT-licensed data.

## ACM LaTeX class

The manuscript uses acmart.cls, copyright 2016–2026 Association for Computing Machinery, maintained by Boris Veytsman. It remains under the LaTeX Project Public License, version 1.3 or any later version, as stated in its upstream source. It is not MIT-licensed. See the [official source](https://github.com/borisveytsman/acmart), [CTAN package record](https://ctan.org/pkg/acmart), and [LPPL text](https://www.latex-project.org/lppl/lppl-1-3c/).

The local class identifies itself as version 2.21, dated 4 October 2026. Matching upstream acmart.dtx, acmart.ins, README, and the LPPL text are supplied in third_party/acmart/. The original source and class notices are preserved. A source-verification record confirms that selecting the upstream docstrip class code produces the same code as the supplied class after ignoring blank lines, full-comment lines, and the generated end-of-input marker. No class code was modified.

The class's existing header requires its original source to accompany distribution of the generated file; the source may be in a different directory. Keep third_party/acmart/ when redistributing the vendored class. Modified class versions must follow the upstream naming and LPPL requirements.

## ACM bibliography style

The supplied ACM-Reference-Format.bst header names Nelson H. F. Beebe, Boris Veytsman, and Gerald Murray and declares its license to be public domain. Its original header is retained. Do not replace that notice with an MIT header.

## Third-party datasets

Provider terms continue to apply to original and derived data files. docs/DATA_SOURCES.md records exact sources, the study's data use, and the verified status of each license.

| Material | Provider terms or verified status |
|---|---|
| Online Retail II | UCI identifies the dataset as CC BY 4.0. Attribute Daqing Chen and the UCI Machine Learning Repository, link the license, and identify preprocessing. |
| Chicago Food Inspections and city boundary | The official metadata states “See Terms of Use.” The City of Chicago Data Terms govern these files and require the derivative-application notice below. |
| BTS Reporting Carrier On-Time Performance | Official federal data source. The supplied project describes it as U.S. public-domain data; this audit did not retrieve a separate dataset-specific license declaration. Preserve BTS provenance and do not describe the input files as MIT-licensed. |
| Raha Tax and Beers dirty/clean pairs | The upstream Raha repository declares Apache License 2.0. A copy of its license is supplied in third_party/raha/. A distinct upstream rights statement for each underlying dataset was not located; the repository license is not represented as a new MIT grant or a verification of every underlying dataset's provenance. |

### Required City of Chicago derivative-application notice

The following notice is reproduced from the City's [Data Terms of Use](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html), also published on an [official City application disclaimer page](https://webapps1.chicago.gov/ChicagoTif/disclaimer.html). Preserve it on the repository or release page where the application or a Chicago-derived data archive can be downloaded:

> This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago. The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site. The data provided at this site is subject to change at any time. It is understood that the data provided at this site is being used at one's own risk.

Additional terms imposed by the supplying City agency continue to apply. The City retains the ability described in its terms to require termination of data display, distribution, or use. The MIT license does not remove those conditions.

## Runtime software and external validators

Python libraries, Deequ, TensorFlow Data Validation, Java, Docker, and other separately installed tools retain their own licenses. Their names identify dependencies or evaluated configurations and do not imply endorsement. This repository does not relicense them. Preserve notices for any dependency files or binaries added to a future release.

## Model services and cached artifacts

Model and service names identify the experimental endpoints recorded by the project. Provider trademarks and service terms remain separate from this repository's license. Cached prompts and responses are research artifacts, not copies of provider model weights. Credentials are not required to inspect the saved results; making new service calls requires the user's own account and applicable provider terms.

## Distribution records

third_party/upstream_sources.json records download URLs, file sizes, and SHA-256 values of the license/source files obtained for this release. The Chicago metadata and terms snapshots record the material checked at preparation time; the linked provider terms are authoritative for subsequent use.
