from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging
import shutil

import numpy as np
import pandas as pd

from .config import BuildConfig, PREFIX_TO_NAMESPACE
from .ontology import Ontology
from .parsers import iter_fasta, iter_uniprot

LOGGER = logging.getLogger(__name__)

DEEPGOPLUS_PICKLE_FILES = {
    "train_data": "train_data.pkl",
    "train_data_train": "train_data_train.pkl",
    "train_data_valid": "train_data_valid.pkl",
    "test_data": "test_data.pkl",
    "terms": "terms.pkl",
}


def load_proteins(paths: tuple[Path, ...], target_taxa: frozenset[str], reviewed_only: bool) -> dict[str, str]:
    proteins: dict[str, str] = {}
    for path in paths:
        LOGGER.info("Loading UniProt sequences from %s", path)
        for rec in iter_uniprot(path):
            if target_taxa and rec.taxon_id not in target_taxa:
                continue
            if reviewed_only and rec.reviewed is False:
                continue
            proteins[rec.protein_id] = rec.sequence
            for accession in rec.accessions:
                proteins.setdefault(accession, rec.sequence)
    LOGGER.info("Loaded %d unique sequence IDs", len(proteins))
    return proteins


def propagate_annotations(go: Ontology, annotations: set[str]) -> set[str]:
    annots_set = set()
    for go_id in annotations:
        annots_set |= go.get_anchestors(go_id)
    return annots_set


def build_training_dataframe(
    go: Ontology,
    sequences: dict[str, str],
    train_annots: dict[str, set[str]],
) -> tuple[pd.DataFrame, Counter]:
    proteins = []
    seqs = []
    prop_annotations = []
    cnt = Counter()
    skipped_missing_sequence = 0
    skipped_empty_propagation = 0

    for prot_id in sorted(train_annots):
        annots = train_annots[prot_id]
        if prot_id not in sequences:
            skipped_missing_sequence += 1
            continue
        annots_set = propagate_annotations(go, annots)
        if not annots_set:
            skipped_empty_propagation += 1
            continue
        proteins.append(prot_id)
        seqs.append(sequences[prot_id])
        prop_annotations.append(tuple(sorted(annots_set)))
        for go_id in sorted(annots_set):
            cnt[go_id] += 1

    df = pd.DataFrame({
        "proteins": proteins,
        "sequences": seqs,
        "annotations": prop_annotations,
    })
    if skipped_missing_sequence:
        LOGGER.warning(
            "Dropped %d t0-annotated proteins from training because no t0 sequence was loaded",
            skipped_missing_sequence,
        )
    if skipped_empty_propagation:
        LOGGER.warning(
            "Dropped %d t0-annotated proteins from training because annotation propagation returned no GO terms",
            skipped_empty_propagation,
        )
    return df, cnt


def build_test_dataframe(
    go: Ontology,
    sequences: dict[str, str],
    t0_annots: dict[str, set[str]],
    t1_annots: dict[str, set[str]],
    train_proteins: set[str],
) -> pd.DataFrame:
    proteins = []
    seqs = []
    prop_annotations = []
    skipped_train_id = 0
    skipped_missing_sequence = 0
    skipped_no_gain = 0
    skipped_empty_propagation = 0

    for prot_id, annots_t1 in t1_annots.items():
        if prot_id in train_proteins:
            skipped_train_id += 1
            continue
        if prot_id not in sequences:
            skipped_missing_sequence += 1
            continue
        gained = annots_t1 - t0_annots.get(prot_id, set())
        if not gained:
            skipped_no_gain += 1
            continue
        annots_set = propagate_annotations(go, gained)
        if not annots_set:
            skipped_empty_propagation += 1
            continue
        proteins.append(prot_id)
        seqs.append(sequences[prot_id])
        prop_annotations.append(annots_set)

    df = pd.DataFrame({
        "proteins": proteins,
        "sequences": seqs,
        "annotations": prop_annotations,
    })
    LOGGER.info(
        "Test candidate filtering: kept=%d skipped_train_or_valid_id=%d "
        "skipped_missing_t1_sequence=%d skipped_no_gained_terms=%d skipped_empty_propagation=%d",
        len(df),
        skipped_train_id,
        skipped_missing_sequence,
        skipped_no_gain,
        skipped_empty_propagation,
    )
    return df


def make_terms_dataframe(cnt: Counter, min_count: int) -> pd.DataFrame:
    # Match DeepGOPlus filtering of propagated training terms.
    terms = []
    for key, val in sorted(cnt.items()):
        if val >= min_count:
            terms.append(key)
    return pd.DataFrame({"terms": terms})


def make_deepgoplus_terms_dataframe(cnt: Counter, min_count: int) -> pd.DataFrame:
    """Preserve DeepGOPlus term filtering and prefix-group ordering."""
    res: dict[str, list[str]] = {}
    for key, val in cnt.items():
        if val >= min_count:
            ont = key.split(":")[0]
            if ont not in res:
                res[ont] = []
            res[ont].append(key)

    terms = []
    for _, val in res.items():
        terms += val
    return pd.DataFrame({"terms": terms})


def load_deepgoplus_annotation_file(path: Path) -> dict[str, set[str]]:
    """Read DeepGOPlus/CAFA two-column tab annotation files."""
    annots: dict[str, set[str]] = {}
    with open(path, "r") as handle:
        for line in handle:
            prot_id, go_id, *_ = line.strip().split("\t")
            if prot_id not in annots:
                annots[prot_id] = set()
            annots[prot_id].add(go_id)
    return annots


def build_deepgoplus_dataframe_from_fasta(
    go: Ontology,
    sequences_file: Path,
    annots: dict[str, set[str]],
    count_terms: bool,
) -> tuple[pd.DataFrame, Counter]:
    """Build a DeepGOPlus-style dataframe from FASTA records and annotations."""
    proteins = []
    sequences = []
    annotations = []
    cnt = Counter()

    for prot_info, sequence in iter_fasta(sequences_file):
        prot_id = prot_info.split()[0]
        if prot_id in annots:
            proteins.append(prot_id)
            sequences.append(sequence)
            annotations.append(annots[prot_id])

    prop_annotations = []
    for direct_annots in annotations:
        annots_set = set()
        for go_id in direct_annots:
            annots_set |= go.get_anchestors(go_id)
        prop_annotations.append(annots_set)
        if count_terms:
            for go_id in annots_set:
                cnt[go_id] += 1

    df = pd.DataFrame({
        "proteins": proteins,
        "sequences": sequences,
        "annotations": prop_annotations,
    })
    return df, cnt


def split_train_valid(df: pd.DataFrame, split: float = 0.9, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Preserve the seeded DeepGOPlus row split.
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    np.random.seed(seed=seed)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    return train_df, valid_df


def terms_for_namespace(go: Ontology, terms_df: pd.DataFrame, namespace: str) -> list[str]:
    terms = []
    for term in terms_df.terms.tolist():
        if term in go.ont and go.get_namespace(term) == namespace:
            terms.append(term)
    return terms


def preprocess_for_terms(go: Ontology, df: pd.DataFrame, list_terms: list[str]) -> pd.DataFrame:
    # Convert annotations to the TEMPROT wide-label layout.
    f_df = {}
    f_df["proteins"] = []
    f_df["sequences"] = []
    term_index = {term: idx for idx, term in enumerate(list_terms)}

    for i in list_terms:
        f_df[i] = []

    for i in range(len(df)):
        is_in_dataset = False
        actual_terms = [0 for _ in range(len(list_terms))]
        protein, sequence, annotation = df.iloc[i, :3].values

        for term in list(annotation):
            if term in term_index:
                is_in_dataset = True
                actual_terms[term_index[term]] = 1
                for ant in go.get_anchestors(term):
                    if ant in term_index:
                        actual_terms[term_index[ant]] = 1

        if is_in_dataset:
            f_df["proteins"].append(protein)
            f_df["sequences"].append(sequence)

            for j in range(len(list_terms)):
                f_df[list_terms[j]].append(actual_terms[j])

    return pd.DataFrame(f_df)


def prepare_pfp_csv_frames(
    go: Ontology,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    terms_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the nine PFP tables once, before they are written."""
    prepared: dict[str, pd.DataFrame] = {}
    for prefix, namespace in PREFIX_TO_NAMESPACE.items():
        list_terms = terms_for_namespace(go, terms_df, namespace)
        LOGGER.info("Exporting %s with %d GO terms", prefix, len(list_terms))

        n_train = preprocess_for_terms(go, train_df, list_terms)
        n_val = preprocess_for_terms(go, valid_df, list_terms)
        n_test = preprocess_for_terms(go, test_df, list_terms)

        # Remove duplicate sequences in test, training, validation order.
        if not n_test.empty:
            n_train = n_train[~n_train.sequences.isin(n_test.sequences.values)]
            n_val = n_val[~n_val.sequences.isin(n_test.sequences.values)]
        if not n_train.empty:
            n_val = n_val[~n_val.sequences.isin(n_train.sequences.values)]

        for split, split_df in {
            "training": n_train,
            "validation": n_val,
            "test": n_test,
        }.items():
            prepared[f"{prefix}-{split}"] = split_df
    return prepared


def write_pfp_csv_frames(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, frame in frames.items():
        path = output_dir / f"{key}.csv"
        frame.to_csv(path, index=False)
        written[key] = path
    return written


def export_pfp_csvs(
    go: Ontology,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    return write_pfp_csv_frames(
        prepare_pfp_csv_frames(go, train_df, valid_df, test_df, terms_df),
        output_dir,
    )


def export_from_deepgoplus_pickles(
    deepgoplus_dir: Path,
    go_obo: Path,
    output_dir: Path,
    include_rels: bool = True,
    write_intermediates: bool = True,
) -> dict[str, Path]:
    """Export PFP-compatible CSVs from released DeepGOPlus/TEMPROT pickles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Loading GO ontology from %s", go_obo)
    go = Ontology(go_obo, with_rels=include_rels)

    paths = {
        key: deepgoplus_dir / filename
        for key, filename in DEEPGOPLUS_PICKLE_FILES.items()
    }

    LOGGER.info("Loading DeepGOPlus train split from %s", paths["train_data_train"])
    train_df = pd.read_pickle(paths["train_data_train"])
    LOGGER.info("Loading DeepGOPlus validation split from %s", paths["train_data_valid"])
    valid_df = pd.read_pickle(paths["train_data_valid"])
    LOGGER.info("Loading DeepGOPlus test data from %s", paths["test_data"])
    test_df = pd.read_pickle(paths["test_data"])
    LOGGER.info("Loading DeepGOPlus terms from %s", paths["terms"])
    terms_df = pd.read_pickle(paths["terms"])

    LOGGER.info(
        "Loaded DeepGOPlus dataframes: train=%s valid=%s test=%s terms=%s",
        train_df.shape,
        valid_df.shape,
        test_df.shape,
        terms_df.shape,
    )

    written: dict[str, Path] = {}
    if write_intermediates:
        for key, src in paths.items():
            dest = output_dir / src.name
            shutil.copy2(src, dest)
            written[key] = dest

    csv_paths = export_pfp_csvs(go, train_df, valid_df.iloc[:, :3], test_df, terms_df, output_dir)
    written.update(csv_paths)
    return written


def generate_deepgoplus_pickles_from_cafa_files(
    go_obo: Path,
    train_sequences_file: Path,
    train_annotations_file: Path,
    test_sequences_file: Path,
    test_annotations_file: Path,
    output_dir: Path,
    min_count: int = 50,
    split: float = 0.9,
    seed: int = 0,
    include_rels: bool = True,
) -> dict[str, Path]:
    """Build DeepGOPlus pickles and PFP CSVs from official CAFA files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Loading GO from %s", go_obo)
    go = Ontology(go_obo, with_rels=include_rels)

    LOGGER.info("Loading training annotations from %s", train_annotations_file)
    train_annots = load_deepgoplus_annotation_file(train_annotations_file)

    LOGGER.info("Loading training sequences from %s", train_sequences_file)
    train_df, cnt = build_deepgoplus_dataframe_from_fasta(
        go=go,
        sequences_file=train_sequences_file,
        annots=train_annots,
        count_terms=True,
    )
    LOGGER.info("Train proteins: %d", len(train_df))

    terms_df = make_deepgoplus_terms_dataframe(cnt, min_count)
    LOGGER.info("Number of terms %d", len(terms_df))

    LOGGER.info("Loading testing annotations from %s", test_annotations_file)
    test_annots = load_deepgoplus_annotation_file(test_annotations_file)

    LOGGER.info("Loading testing sequences from %s", test_sequences_file)
    test_df, _ = build_deepgoplus_dataframe_from_fasta(
        go=go,
        sequences_file=test_sequences_file,
        annots=test_annots,
        count_terms=False,
    )
    LOGGER.info("Test proteins: %d", len(test_df))

    written = {
        "train_data": output_dir / "train_data.pkl",
        "test_data": output_dir / "test_data.pkl",
        "terms": output_dir / "terms.pkl",
    }
    LOGGER.info("Saving training data to %s", written["train_data"])
    train_df.to_pickle(written["train_data"])
    LOGGER.info("Saving terms to %s", written["terms"])
    terms_df.to_pickle(written["terms"])
    LOGGER.info("Saving testing data to %s", written["test_data"])
    test_df.to_pickle(written["test_data"])

    train_split_df, valid_split_df = split_train_valid(train_df, split=split, seed=seed)
    written.update(
        export_pfp_csvs(
            go=go,
            train_df=train_split_df,
            valid_df=valid_split_df,
            test_df=test_df,
            terms_df=terms_df,
            output_dir=output_dir,
        )
    )
    return written


def build_benchmark(config: BuildConfig) -> dict[str, Path]:
    # Load the snapshot route only when requested.
    from .snapshot import build_snapshot_benchmark

    return build_snapshot_benchmark(config)
