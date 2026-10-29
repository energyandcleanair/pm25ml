### Data versioning

For each release, we provide versioning information to ensure every filename identifies when it was released, which schema it follows, and which model generated it. Major versions indicate breaking or significant changes in the schema or model.

Filename pattern: `pm25-<granule>_<rel>_<s>_<mb>.nc`

### `<granule>`: spatial--temporal packaging

The file's spatial--temporal packaging.

Format: lowercase letters and hyphens only. Options available: `full` only.

### `<rel>`: release

Release month of the file and regeneration version. Independent of `s` and `mb`. This does not describe the maximum date available in the dataset, which is usually delayed by up to 2 months. 

Format: `rel-YYYY-MM-rT`

-   `YYYY`: four-digit year.
-   `MM`: two-digit month.
-   `T`: zero-based monthly regeneration counter. Resets to `0` when `YYYY-MM` changes.

We may withhold a release if the results do not meet our quality requirements or the dataset does not cover the expected time period. Consequently, published releases may have gaps in the `rT` sequence.

### `<s>`: schema

The file's schema.

Format: `s-X.Y`

-   `X`: major schema version. Breaking or significant changes in file schema or variable layout.
-   `Y`: minor/patch schema version. Backward-compatible metadata or layout additions.

### `<mb>`: model bundle

The model bundle used to generate the dataset. This includes the models at every stage, their features, ingestion, and model code. Changes to any of these can change the model bundle version.

The model bundle uses semantic versioning to distinguish significant methodological changes, improvements, and fixes. These versions describe changes to the modelling system and the interpretation of its results.

Format: `mb-X.Y.Z`

-   `X`: major model version. Changes to model features or modelling approach that significantly affect the methodology or comparability of results.
-   `Y`: minor model version. Backward-compatible improvements to the modelling system that preserve the feature set and overall modelling approach, but may change the results.
-   `Z`: patch model version. Bug fixes or maintenance changes that preserve the intended methodology and feature set. Corrections may still affect the results.

### Change to version numbering

Previously, model bundles used the format `mb-X.Y-rT`, where `T` was a retraining counter. However, retraining was captured by `rel`.

This has been replaced by `mb-X.Y.Z` to separate changes to the modelling system from individual retraining runs.

The new `Z` component represents a patch version

Earlier model bundle versions already followed the same major and minor versioning principles.