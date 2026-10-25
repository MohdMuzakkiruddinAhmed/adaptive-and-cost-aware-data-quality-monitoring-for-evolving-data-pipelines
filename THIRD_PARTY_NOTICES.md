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
