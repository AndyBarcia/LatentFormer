import warnings
from importlib import util
from pathlib import Path


_NATIVE_OP = None
_IMPORT_ERROR = None
_WARNED = False


def _load_native_op():
    global _NATIVE_OP, _IMPORT_ERROR
    if _NATIVE_OP is not None or _IMPORT_ERROR is not None:
        return _NATIVE_OP
    try:
        import LatentFormerSeedSelection as native_op
    except ModuleNotFoundError as exc:
        native_path = next(Path(__file__).resolve().parents[1].glob("LatentFormerSeedSelection*.so"), None)
        if native_path is None:
            _IMPORT_ERROR = exc
            return None
        spec = util.spec_from_file_location("LatentFormerSeedSelection", native_path)
        if spec is None or spec.loader is None:
            _IMPORT_ERROR = exc
            return None
        native_op = util.module_from_spec(spec)
        spec.loader.exec_module(native_op)
    _NATIVE_OP = native_op
    return _NATIVE_OP


def clustering_seed_selection_native(
    query_signatures,
    query_seed_logits,
    *,
    seed_threshold,
    duplicate_threshold,
    similarity_metric,
    eps=1e-6,
    temp=0.1,
):
    native_op = _load_native_op()
    if native_op is None:
        global _WARNED
        if not _WARNED:
            warnings.warn(
                "LatentFormerSeedSelection extension is not built; falling back to the "
                "Python ClusteringSeedSelection implementation. Build it with "
                "`cd mask2former/modeling/seed_selection_ops && sh make.sh`.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED = True
        return None

    return native_op.clustering_seed_selection_forward(
        query_signatures,
        query_seed_logits,
        float(seed_threshold),
        float(duplicate_threshold),
        similarity_metric.lower(),
        float(eps),
        float(temp),
    )
