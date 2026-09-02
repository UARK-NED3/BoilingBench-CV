# BoilingBench-CV v0.1 protocol

## Scope

v0.1 evaluates one semantic class, `bubble`, as separate two-dimensional instances. A polygon denotes the visible projected interface region labeled by the original annotator. It does not establish a three-dimensional bubble volume, void fraction, departure diameter, or temporal identity.

## Unit of independence

The split unit is a source acquisition group, inferred as `(regime, source_video)`. All frames from that group must be assigned to one partition. Crops, resizes, augmented variants, pseudo-labels, and derived masks inherit the same group.

## Annotation conversion

The canonical schema follows COCO instance segmentation conventions: one image record with dimensions and metadata; one annotation per bubble with its original-coordinate polygon, bounding box, and shoelace area; and an `extra` object retaining source provenance. Conversion does not simplify, smooth, merge, or repair contours. Geometry problems are recorded for human adjudication.

## Required run record

Each run must retain benchmark and split hashes; model source revision, license and checkpoint hash; package versions; operating system; accelerator; preprocessing; command; seed; elapsed time; stratum-level metrics; and representative failures from each regime.

## Dynamics extension gate

Tracking metrics and bubble-event metrics are out of scope until contiguous clips have reviewed frame-to-frame identities and a defined event ontology. Per-image contours alone are insufficient evidence for a dynamics claim.
