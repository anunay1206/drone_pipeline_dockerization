# Notes — Understanding the pipeline and thinking about an API

## What the original code does, in plain terms

The existing code is an end-to-end pipeline that takes a drone orthomosaic of a forest and produces, eventually, a species-distribution map you can open in Google Earth. Reading through it, the flow falls into five conceptual stages:

1. **Detection.** A pretrained Detectree2 model finds individual tree crowns in the orthomosaic and outputs each crown as a polygon.
2. **Cropping and feature extraction.** Each crown polygon is cut out of the ortho as its own small image, and a DINOv2 vision model turns each crop into a feature vector that captures what that crown "looks like."
3. **Clustering.** KMeans groups the crowns into clusters at several candidate values of *k*. The code also produces some helper visuals at this point — a t-SNE scatter and a plot of cluster-quality metrics (silhouette, Davies–Bouldin, elbow) — so the person running it can judge which *k* is sensible.
4. **A human step.** The person opens the cluster folders, looks at the sample crown images inside each cluster, decides which *k* to trust, and types a species name next to each cluster number in a CSV.
5. **Assignment, validation, export.** The cluster-to-species mapping is propagated to every crown in the dataset, optionally compared against a small pre-labeled ground-truth set, and the final polygons (each tagged with a species) are exported as a KMZ for Google Earth.

The most important thing we noticed isn't really one of the stages — it's that there is a **mandatory human pause in the middle**, between clustering and species assignment. The pipeline literally cannot continue until a person has looked at the clusters and filled in the species CSV. Everything before the pause is automated, everything after the pause is automated, but the pause itself is the point of the whole pipeline — that's where the model's groupings become meaningful species labels.

We also noticed that the heavy steps (detection, feature extraction, t-SNE) take real time and prefer a GPU, and that every stage writes files on disk that the next stage reads. So the intermediate outputs are real state, not just logs.

## What we're thinking about exposing as an API

Given the human pause and the long runtimes, a single "upload and wait" endpoint isn't workable — a normal HTTP request would time out long before the pipeline finishes, and even if it didn't, the pipeline cannot complete in one shot without the user looking at the clusters first. So the natural shape we keep coming back to is to split the pipeline into **two halves separated by a review step**, with the API tracking where each run currently sits.

The mental model is a **project**: one orthomosaic (or a small set), one configuration, one outcome. A project moves through a few states — uploaded, being analyzed, waiting for labels, finalizing, done — and the API's job is just to let the user push it from one state to the next at their own pace.

The areas where we think it makes sense to have endpoints are:

- **Creating a project and uploading data.** The orthomosaic itself, and optionally a small zip of pre-labeled crowns if the user wants validation. The choice of which Detectree2 weights to use also belongs here.
- **Starting the heavy first half.** Detection plus crown cropping plus features plus clustering, plus the two helper visuals. This is the long-running part, so whatever endpoint kicks it off should return quickly and let the user poll for progress instead of waiting on a single request.
- **Reviewing the clusters.** This is the centrepiece for the user — they need access to the two visualizations (the metric plot and the t-SNE scatter), plus a few sample crown images from each cluster, so they can decide which *k* to trust and what species each cluster represents.
- **Submitting labels.** A simple, structured way to say "I chose k = 6, cluster 0 is acacia, cluster 1 is non-acacia, …" — this is what closes the human gate.
- **Starting the second half.** Species assignment, optional validation against the uploaded ground truth, KMZ export.
- **Getting the results.** Summary numbers (species counts, accuracy if there was ground truth) and the downloadable artifacts — chiefly the KMZ for Google Earth.

## Parameters and metadata we'd want to expose

A second thing the API has to think about, on top of the stages themselves, is what *configuration* the user controls and what *information* the system tracks and returns.

On the input side, a project has a handful of knobs that meaningfully change the result, and we'd expose these rather than bake them in:

- **Which Detectree2 weights to use.** There are three pretrained variants available (urban-trees, tropical UAV, and a general one). Different forests respond differently to each, so this should be a per-project choice with a sensible default.
- **Detection parameters** — tile size, buffer, IoU threshold, confidence threshold. These control how aggressively the detector proposes crowns and how it merges overlapping ones.
- **Clustering parameters** — which candidate *k* values to try, how many PCA components to reduce features to, and the batch size used during feature extraction.
- **Coordinate system (EPSG)** of the orthomosaic — needed because the final KMZ has to be re-projected to WGS84 for Google Earth. We'd try to auto-detect this from the GeoTIFF itself but allow the user to override.

Most of these have sensible defaults from the original code, so a user shouldn't *have* to set any of them for a basic run.

On the output side, there's a second category of information the API needs to track and surface, not just the artifacts at the very end:

- **For each uploaded orthomosaic** — width, height, band count, detected coordinate system. Useful both for confirmation and for auto-filling the EPSG.
- **For each project** — what state it's in (uploaded, analyzing, waiting for labels, finalizing, done, failed), which *k* values are available for review, what the auto-recommended *k* turned out to be, and the cluster→species mapping the user eventually submitted.
- **For each long-running job** — the current sub-stage (detecting, extracting features, clustering, …), a rough progress fraction, the tail of the pipeline's log so the user can see what is happening live, and an error + which stage it occurred in if something fails.
- **For the final results** — per-species crown counts, and validation metrics (accuracy, etc.) if a ground-truth set was uploaded.

This second category is what lets the user understand *what is happening* during a multi-minute run and *what they got* at the end, instead of just receiving a file with no context.

## Open questions we're still thinking through

- Should one project be allowed to hold several orthomosaics (the four survey spots together), or should each spot be its own project? Each option changes what "the result" means.
- How to keep each user's intermediate files isolated when several runs happen at once, since the original code has fairly fixed output paths.
- How to make re-runs clean — if a user wants to re-cluster with different parameters, we don't want old cached features or cluster labels leaking into the new attempt.
- How much progress detail to surface back to the user during the long first half (just the current stage, or finer-grained status).

The overall goal is to keep the human-in-the-loop step (which is what gives the species labels their meaning) while making the rest of the pipeline practical to use as a service rather than as a sequence of command-line scripts.
