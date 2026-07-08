# Subclass Prompt Notes

The semantic subclass classifier receives one target cue and its full sentence context. The model classifies only the target cue and does not assign labels based on other cues in the same sentence.

The output format is a JSON object containing a nonempty `labels` array. Valid labels are the 17 semantic subclass identifiers, `NO`, and `UNMATCHED`. A report-level label is derived after expression-level review and mapping.
