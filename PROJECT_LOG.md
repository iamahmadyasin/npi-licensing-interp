# Project Log of Mechanistic Analysis of NPI Licensing in Gemma-2-2B

A chronological record of design decisions, actions, errors, and course corrections
taken while designing and running this interpretability project. This log is intended
as a research diary. It documents *why* each choice was made, not just *what* was done,
so the reasoning is reconstructable later.

---

## 0. Context

**Research question.** Does Gemma-2-2B represent the *scope* of negation (an abstract
structural relation), or merely its *presence*? Negative-polarity-item (NPI) licensing,
the fact that "any" is grammatical only under an appropriate licensor, is used as the
diagnostic.

**Why this question.** It targets a published gap. Kletz et al. (2024)
used linear probes and could not distinguish whether the model has a genuine negation-scope
representation or a shallower clause-boundary feature, because probes read distributed
signals. Sparse autoencoders (SAEs) can isolate individual features, and causal ablation
can then test them — settling what probing could not.

**Novelty check.** A literature search confirmed that the specific framing
(SAE-based isolation + causal ablation of an NPI-licensing / negation-scope feature,
dissociated from a clause-boundary confound) had not been done.

**Model / tooling choice.** Gemma-2-2B (open, runs on a single free GPU) + GemmaScope
pretrained SAEs (no SAE training required) via `sae_lens`. Raw HuggingFace transformers +
PyTorch hooks were used for interventions rather than TransformerLens.

---

## 1. Experimental Design

**Decision.** Test NPI licensing with controlled minimal-pair triplets:

- **A:** in-scope licensor, grammatical e.g. *"The manager did not approve any proposals."*
- **B:** out-of-scope licensor (trapped in a relative clause), ungrammatical e.g.
  *"The manager who did not approve the budget reviewed any proposals."*
- **C:** no licensor, ungrammatical e.g. *"The manager did approve any proposals."*

**Rationale.**
- **A vs B** is the Kletz et al. (2024) confound and isolates scope specifically. If the
  model only detected "is there a 'not' somewhere," A and B would look alike.
- **B vs C** tests whether the model uses a shallow "negation anywhere" heuristic. If
  B ≈ C, out-of-scope negation behaves like no negation, supporting a true scope
  representation.
- Measurement: log P("any") at the relevant position.

**Design control.** Condition C uses do-support ("did approve") to keep the auxiliary
structure parallel with A ("did not approve"), so the contrast isolates negation rather
than the presence/absence of an auxiliary.

**NPI choice.** Started with both "any" and "ever". "ever" was later dropped.

---

## 2. Initial stimulus set

**Action.** Hand curated 30 items (25 using "any", 5 using "ever"), each a triplet, all in a
single sentential-negation template ("The X did not V any N"). Stored as a CSV.

**Design intent.** Lexical variety across items (different subjects, verbs, objects) but a
fixed syntactic construction, so the controlled contrast is clean. The relative-clause
device in condition B traps the licensor out of scope.

---

## 3. Phase 1: Behavioral validation

**Goal.** Confirm the model even shows the behavioral effect before doing any
interpretability. If it doesn't distinguish the conditions, there is nothing to explain.

**Method.** For each sentence, extract the model's log-probability for the NPI token at its
position, compare across A/B/C, run paired t-tests and effect sizes.

**Errors encountered:**
- **HuggingFace auth / gated repo.** `huggingface-cli` was deprecated; had to accept the
  Gemma license on the model page, create a token, and use `hf auth login`.
- **Tokenizer bug.** Gemma uses SentencePiece; the token for " any" (leading
  space) differs from "any". Matching the NPI by re-encoding "any" and comparing token IDs
  failed for every sentence. **Fix:** locate the NPI by decoding each token, stripping
  whitespace, and string-matching.

**Result (original 25-item "any" set).** A ≫ B ≈ C; very large effect sizes
(d ≈ 1.8 A-vs-B, 2.4 A-vs-C); B vs C non-significant. The model tracks scope, not just
presence. "ever" items were weak and inconsistent and thus dropped.

**Decision.** Proceed to interpretability. Focus on "any" only.

---

## 4. Phase 2: representation at the NPI position

**Goal.** Find SAE features (16k GemmaScope) at the "any" token that separate A from B/C.

**Method.** Extract residual-stream activations at the "any" position across layers, encode
through SAEs, rank features by A-vs-(B+C) separation on a discovery split, validate on a
held-out split. Classify features.

**Result.** Found ~12 SCOPE_SPECIFIC features. Neuronpedia inspection showed they encode
things like the some/any existential class, the n-word/NPI class, and negation salience
i.e. licensing is represented by a distributed constellation of features, not one neuron.

**Conclusion.** Features at the "any" position
represent a licensed NPI once read, but they are downstream of the logit that produces P("any") because of causal attention . They therefore cannot be valid
causal-ablation targets. This notebook was thus
not part of the causal pipeline. Thus it was superseded.

---

## 5. The causal-attention error

**Error.** The first causal-ablation attempt intervened on features at the "any"
token and measured the change in P("any"). Every trial returned exactly zero effect.

**Diagnosis.** Attention is causal. P("any") is computed from the logits at the position
before "any" (the verb). Any intervention at or after the "any" token cannot affect that
logit because the information flows strictly left-to-right. The zero effect was the code
reporting a fact about causal attention.

**Hook propagation.** HuggingFace forward hooks that return a modified
tuple do not reliably replace a decoder layer's output. Even after moving to the correct
position, ablation appeared to do nothing.
- **Fixes:**
  - Intervene via `register_forward_pre_hook` on layer **L+1** (whose input equals layer
    L's output), returning `((new_hidden,) + args[1:], kwargs)` with `with_kwargs=True`.
  - Add `use_cache=False` to the model call (otherwise the measured logit can be served from
    a KV cache built before the intervention).
  - Add a sanity-check assertion: the reconstruction-only pass (route through the SAE,
    ablate nothing) must produce a *nonzero* change in P("any"). If it is zero, the
    intervention is not upstream of the measurement and thus experiment is halted.

**Abandoning TransformerLens.** It doubled memory usage (casts weights to float32
during processing) and caused OOM on the free GPU. Raw HF + PyTorch hooks were used instead.

---

## 6. Redesign

After the causal-attention fix, three upgrades were decided together:

**6a. Move discovery + ablation to the pre-NPI position** (`any_pos - 1`), which is causally
upstream of P("any"). Both intervention and measurement occur there.

**6b. Expand the stimulus set for a generalization test.** Add licensing
environments beyond sentential negation, specifically including environments that license
NPIs *without any negation*, to test whether a discovered feature is an *NPI-licenser* or
merely a *negation-detector*. Final set (80 items, 215 rows):

  - **sentential_negation** 25 items, triplets (A/B/C)
  - **no_subject** 15 items, triplets
  - **few_subject** 15 items, triplets
  - **question** 15 items, A/C pairs
  - **without** 10 items, A/C pairs

  Environments with a clean structural
  out-of-scope counterpart use full triplets while questions and "without" have no coherent
  out-of-scope form, so they run as A/C pairs. The triplets support the
  confirmatory scope analysis; the pairs test licensing generalization. Two
  complementary claims, kept distinct rather than conflated.

**6c. Move from 16k to 65k GemmaScope SAEs.** The 16k reconstruction noise floor
(~0.6–0.9 log-probs) was swallowing weaker features' effects. Wider SAEs reconstruct better,
lowering the floor and raising statistical power. 65k feature indices
are entirely different from 16k, so discovery had to be re-run before any ablation.

**6d. Discovery split decision.** Discover on sentential negation only
(items 1–15 discover, 16–25 validate), and hold all other environments out purely for
generalization testing. This yields the strongest possible generalization claim.
Features found using only "not" sentences, tested on non-negation licensing, with no
opportunity to be tuned to the other environments however this had lightly less
discovery power (15 items), which is immaterial given the very large effect sizes.

---

## 7. Infrastructure and repo hygiene

**Platform decision.** Moved from Colab free (single T4, ~13 GB RAM) to Kaggle for the 65k
runs. 65k SAEs (~1.2 GB each) plus the model exceed comfortable Colab-free memory.
Kaggle offers more RAM and T4×2 option. On T4×2, `device_map="auto"` split the model
across two GPUs and caused a cuda:0/cuda:1 device-mismatch in the hooks; fixed by pinning the
whole model and SAEs to a single device (`device_map={"": "cuda:0"}`). SAEs loaded on the same device as the model.

**Repo hygiene decisions.**
- **Single source of truth for stimuli.** Stimuli moved out of hardcoded notebook lists into
  one CSV (`stimuli/npi_stimuli.csv`) loaded by every notebook. This prevents the
  silent drift that earlier caused an "ever"/"any" inconsistency between notebooks.
- **Robust file loading on Kaggle.** Used `glob.glob("/kaggle/input/**/npi_stimuli.csv")`
  to find the dataset regardless of the exact mount path (Kaggle nested it under
  `datasets/<user>/<slug>/`).
- **What is kept.** The GitHub repo holds only the clean pipeline. The archived
  representation notebook itself and its results (tables and images)
  were not pushed to git. The invalid any-position ablation outputs (the all-zeros CSV and flat plot) were
  kept as evidence for the methods lesson.
- **Consistency conventions carried across notebooks:** align paired statistics by pivoting
  on `item` (never by pulling `.values` off separately-filtered frames); A-vs-B analyses on
  triplet items only, A-vs-C on all items; per-environment breakdowns reported as results;
  outputs named `0N_<stage>_<what>` and written to `/kaggle/working/`; imports consolidated;
  no leftover "ever" or hardcoded item-range labels.

---

## 8. Discovery run at the pre-NPI position (65k)

**Verification.** `SAE.get_saes_directory` did not exist in the installed sae_lens version;
resolved by loading the 65k SAE directly. Confirmed load: shape [215, 65536], healthy L0
(avg ~75–128 active features), canonical variants `average_l0_72`–`average_l0_128`.

**pattern classification:** SCOPE_SPECIFIC 29,
NEGATION_GRADED 2, PARTIAL 14, WEAK 19, INVERSE 56. Discovery t-statistics up to ~35
(cleaner than the 16k run's ~20), confirming the width upgrade improved separation.

**Incidental observation.** A large INVERSE set (56 features firing in B/C, near-zero in A)
appeared containing features positively representing the unlicensed condition, not merely the
absence of a licensing feature. Flagged for the optional reverse test.

**Generalization table.** Top general features, discovered on sentential
negation only, showed strong A-vs-C separation across all environments including
negation-free ones (question, without) such as L20 F31666 gaps: sent.neg 54.0, no 45.3,
few 34.8, question 24.1, without 46.9. In contrast, L14 F51903 separated well under
negation and "without" but collapsed to ~0 in no/few/question.

---

## 9. Ablation run at the pre-NPI position (65k)

**Target selection (from validated discovery):**
- `L22_F6415_general`, `L20_F31666_general` as strongest, broadest generalizers.
- `L14_F51903_negspec` as negation-specific; included deliberately to test dissociation.
- `CONTROL_L22_F22498` as a WEAK-pattern feature that fires in A (~7) but is not
  licensing-selective; chosen as a negative control (rules out "ablating any active feature
  hurts licensing"). A near-null but marginal feature was preferred over a fully dead one so
  it controls the stronger alternative hypothesis.
- `COMBINED` top three generalizers, to test distributed coding.

**Statistical-design decisions:**
- Effect computed as `recon − ablate`, so the SAE reconstruction artifact cancels (both
  terms route through the SAE). Significance judged against the control feature and
  against zero (selectivity t-test), not against the raw `baseline − recon`
  artifact. The two use different reference points and conflating them would misstate
  results.
- Per-environment breakdown reported as the headline result.
- Memory guard added for resident 65k SAEs; single-layer fallback documented if OOM.

**Results.**
- General features: large, selective A-effects (F31666 ≈ 2.69, F6415 ≈ 1.44; COMBINED
  ≈ 3.15), all well above the 65k noise floor (~0.48–0.72) and ~50–100× the control.
  Selectivity d = 1.4–2.0, p < 1e-6.
- Generalization: effects positive in every environment including question and without.
- Dissociation: F51903 had an effect under negation (+0.32) but exactly zero in
  no/few/question, cleanly dissociating from the general features.
- Control: ≈ +0.03 overall, negligible everywhere.
- Environment gradient: effects strongest for few/no, weakest for questions.

---

## 10. Optional reverse-ablation test

**Hypothesis.** If licensed-features causally raise P("any"), do features that fire in the
unlicensed condition (C) causally suppress it? Prediction: ablating them in C should
raise P("any") (a negative effect under the `recon − ablate` convention).

**Method.** Ablated four clean INVERSE features (fire in C, ~0 in A/B). Freed the two
unneeded resident SAEs first to fit memory on a single T4.

**Result.** Prediction not borne out. Effects in C were small, positive (P("any")
fell slightly further), decayed with depth, and vanished when combined (t = 0.46, p = 0.64).
Effects in A/B were exactly zero (built-in control passed).

**Interpretation.** NPI licensing is causally asymmetric: the model maintains a positive,
causally-efficacious representation of licensed NPIs, but no releasable "suppressor" of
unlicensed ones. Neuronpedia inspection showed the strongest "unlicensed" feature is
actually a document-structure / sentence-initial feature that merely correlated with
condition C's short declaratives explaining the null cleanly. A more sophisticated finding
than a tidy symmetry, and only surfaced because the obvious alternative was tested.

---

## 11. Neuronpedia

Inspected top-activating examples and positive/negative logits for each final feature
(auto-generated labels treated as unreliable hints; activations/logits treated as evidence).

- **L22 F6415 / L20 F31666 (general).** Positive logits dominated by polarity items
  (anything, anymore, anywhere, anyone, nor); activations span negation, "without", "hardly",
  interrogatives. Natural behavior matches broad causal generalization.
- **L14 F51903 (negation-specific).** Activations concentrate on verbs immediately following
  explicit "not" (does not X, did not X). Explains its failure to generalize.
- **L22 F22498 (control).** Fires on formulaic temporal "any" (at any time/point, anytime);
  no NPIs in positive logits thus a different lexical "any". Explains its null effect.
- **L12 F56118 (unlicensed).** Fires predominantly on `<bos>` and sentence/section-initial
  positions thus a structural feature, not an NPI suppressor. Explains the asymmetry.

**Value.** The qualitative semantics converge with the causal results at every point,
arguing the effects are not SAE-decomposition artifacts and the features mean what the
experiment says they mean.

---

## 12. Summary

Behavior tracks scope (with a secondary, quantifier-driven residual-presence effect) →
SAE features at the pre-NPI position represent licensing → ablating them causally reduces
P("any"), far above noise floor and control → the effect holds in negation-free environments
(abstract licensing, not negation detection) → general vs negation-specific features doubly
dissociate → licensing is distributed/redundant (COMBINED largest) → licensing is causally
asymmetric (no releasable suppressor) → qualitative feature semantics independently
corroborate the causal roles. Plus two methods contributions: (i) interventions must be
causally upstream of the measured token; (ii) SAE width materially affects the detectable
noise floor and thus statistical power.

---

## 13. Quick reference

| Decision | Choice | Reason |
|---|---|---|
| Diagnostic phenomenon | NPI licensing ("any") | Clean minimal pairs; targets Kletz et al. gap |
| Conditions | A/B/C triplets | A-vs-B isolates scope; B-vs-C tests shallow heuristic |
| NPI | "any" only | "ever" too weak/inconsistent |
| Intervention position | pre-NPI (`any_pos−1`) | Causal attention: must be upstream of P("any") |
| Hook method | pre-hook on L+1, `use_cache=False` | Reliable output replacement; avoid KV-cache staleness |
| SAE width | 65k (from 16k) | Lower reconstruction noise floor → more power |
| Discovery split | sentential negation only | Strongest generalization claim |
| Extra environments | no/few/question/without | Test NPI-licenser vs negation-detector |
| Triplet vs pair | mixed | Pairs where no clean out-of-scope form exists |
| Effect baseline | `recon − ablate` | Cancels SAE reconstruction artifact |
| Significance bar | vs control + vs zero | Not vs raw reconstruction artifact |
| Control feature | active but non-selective (WEAK) | Rules out "any active feature" artifact |
| Platform | Kaggle, single GPU pinned | Memory for 65k SAEs; avoid multi-GPU device mismatch |
| Stimuli storage | one CSV, glob-loaded | Single source of truth; prevents notebook drift |
| Reverse test | ablate INVERSE features | Tests causal symmetry; found asymmetry |
| Feature labels | activations/logits over auto-labels | Auto-labels unreliable |
