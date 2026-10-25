"""Verify ground-truth labels against the FINAL batch (manuscript Sections 6.2 and 8.2).

Individual injectors can each be correct while their combination is not: a semantic mutation of the
original record of an injected duplicate pair left its copy labelled "duplicate" although it no
longer duplicated anything (70 of 6,750 labels in the proof-of-concept semantic batches). Labels
are therefore checked after every mutation, against the batch as it is finally written.

``audit_batch`` compares the final batch with the reference batch from which it was built (the
clean base batch after legitimate drift, before any defect) and checks:

  duplicate labels   every record labelled "duplicate" is an exact whole-record copy (all columns
                     except record_id) of an earlier record in the final batch
  unlabelled records every unlabelled original record is unchanged from the reference (except for
                     declared legitimate changes, e.g. injected valid category novelty) and is not
                     itself a whole-record duplicate of an earlier record
  labelled records   every non-duplicate defect changed its labelled column (NaN for "missing")
  identifiers        every record is either a reference record or a labelled duplicate; record IDs
                     are unique; every label points to a record of the batch

The result lists every inconsistency; ``strict=True`` raises instead.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


class LabelAuditError(AssertionError):
    pass


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality that treats NaN == NaN and compares numbers with a tolerance."""
    an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    both_num = an.notna().to_numpy() & bn.notna().to_numpy()
    eq = np.zeros(len(a), dtype=bool)
    eq[both_num] = np.isclose(an.to_numpy()[both_num], bn.to_numpy()[both_num], rtol=1e-12, atol=1e-12)
    rest = ~both_num
    sa, sb = a.to_numpy(dtype=object)[rest], b.to_numpy(dtype=object)[rest]
    na_a, na_b = pd.isna(sa), pd.isna(sb)
    eq[rest] = (na_a & na_b) | (~na_a & ~na_b & (sa.astype(str) == sb.astype(str)))
    return eq


def audit_batch(final: pd.DataFrame, gt: pd.DataFrame, reference: pd.DataFrame,
                legitimate_changes: Optional[Mapping[str, Iterable[str]]] = None,
                strict: bool = False, max_examples: int = 10) -> Dict[str, Any]:
    """Check the labels ``gt`` of batch ``final`` (see module docstring).

    ``legitimate_changes`` maps record_id -> columns that were changed on purpose without being a
    defect (valid novelty). Returns {"ok": bool, "n_labels": .., "issues": {kind: count}, "examples": [...]}.
    """
    legit = {k: set(v) for k, v in (legitimate_changes or {}).items()}
    issues: Dict[str, int] = {}
    examples: List[str] = []

    def bad(kind: str, ids: Iterable[str]) -> None:
        ids = list(ids)
        if ids:
            issues[kind] = issues.get(kind, 0) + len(ids)
            for r in ids[: max(0, max_examples - len(examples))]:
                examples.append(f"{kind}: {r}")

    cols = [c for c in final.columns if c != "record_id"]
    ids = final["record_id"].astype(str)
    bad("duplicate_record_id", ids[ids.duplicated()].tolist())
    labelled = gt.groupby("record_id")["defect_type"].agg(list).to_dict()
    bad("label_without_record", sorted(set(labelled) - set(ids)))
    bad("record_with_several_labels", [r for r, v in labelled.items() if len(v) > 1])

    dup_ids = set(gt.loc[gt["defect_type"] == "duplicate", "record_id"].astype(str))
    ref_ids = set(reference["record_id"].astype(str))
    bad("unknown_record", sorted(set(ids) - ref_ids - dup_ids))

    # duplicate labels: exact whole-record copy of an EARLIER record
    is_dup = final.duplicated(subset=cols, keep="first").to_numpy()
    pos = {r: i for i, r in enumerate(ids)}
    bad("duplicate_label_not_a_duplicate", sorted(r for r in dup_ids if r in pos and not is_dup[pos[r]]))
    # an unlabelled record that duplicates an earlier record is an unlabelled defect
    unl_dup = [r for r, d in zip(ids, is_dup) if d and r not in labelled]
    bad("unlabelled_duplicate", unl_dup)

    # compare original records with the reference
    orig = final[ids.isin(ref_ids).to_numpy()].set_index(final["record_id"].astype(str)[ids.isin(ref_ids).to_numpy()])
    ref = reference.set_index(reference["record_id"].astype(str)).loc[orig.index]
    changed: Dict[str, set] = {r: set() for r in orig.index}
    for c in cols:
        if c not in ref.columns:
            continue
        diff = ~_same(orig[c], ref[c])
        for r in orig.index[diff]:
            changed[r].add(c)
    unlabelled_changed, label_no_change, wrong_column = [], [], []
    gt_cols = gt.set_index(gt["record_id"].astype(str))
    for r, ch in changed.items():
        ch = ch - legit.get(r, set())
        if r not in labelled:
            if ch:
                unlabelled_changed.append(r)
            continue
        kinds = labelled[r]
        if kinds == ["duplicate"]:
            continue
        lab_col = gt_cols.loc[r, "column"]
        lab_col = lab_col.iloc[0] if isinstance(lab_col, pd.Series) else lab_col
        if lab_col == "<row>":
            continue
        if not ch:
            label_no_change.append(r)
        elif ch - {lab_col}:
            wrong_column.append(r)
        if "missing" in kinds and not pd.isna(orig.at[r, lab_col]):
            label_no_change.append(r)
    bad("unlabelled_record_changed", unlabelled_changed)
    bad("labelled_defect_left_value_unchanged", label_no_change)
    bad("defect_changed_other_columns", wrong_column)

    out = {"ok": not issues, "n_labels": int(len(gt)), "n_records": int(len(final)),
           "issues": issues, "examples": examples}
    if strict and issues:
        raise LabelAuditError(f"label audit failed: {issues}; e.g. {examples[:3]}")
    return out
