import logging
from contextlib import contextmanager

import os
import os.path as op
import sys

import numpy as np


class Bunch(dict):
    """A dict with additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return Bunch(super(Bunch, self).copy())


def _pad(arr, n, dir='right'):
    """Pad an array with zeros along the first axis.

    Parameters
    ----------

    n : int
        Size of the returned array in the first axis.
    dir : str
        Direction of the padding. Must be one 'left' or 'right'.

    """
    assert dir in ('left', 'right')
    if n < 0:
        raise ValueError("'n' must be positive: {0}.".format(n))
    elif n == 0:
        return np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
    n_arr = arr.shape[0]
    shape = (n,) + arr.shape[1:]
    if n_arr == n:
        assert arr.shape == shape
        return arr
    elif n_arr < n:
        out = np.zeros(shape, dtype=arr.dtype)
        if dir == 'left':
            out[-n_arr:, ...] = arr
        elif dir == 'right':
            out[:n_arr, ...] = arr
        assert out.shape == shape
        return out
    else:
        if dir == 'left':
            out = arr[-n:, ...]
        elif dir == 'right':
            out = arr[:n, ...]
        assert out.shape == shape
        return out


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _is_integer(x):
    return isinstance(x, integer_types + (np.generic,))


def _as_tuple(item):
    """Ensure an item is a tuple."""
    if item is None:
        return None
    elif not isinstance(item, tuple):
        return (item,)
    else:
        return item


def _spikes_per_cluster(spike_ids, spike_clusters):
    """Return a dictionary {cluster: list_of_spikes}."""
    if not len(spike_ids):
        return {}
    rel_spikes = np.argsort(spike_clusters)
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    spikes_in_clusters = {clusters[i]: np.sort(abs_spikes[idx[i]:idx[i + 1]])
                          for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = np.sort(abs_spikes[idx[-1]:])

    return spikes_in_clusters


def _read_python(path):
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = np.asarray(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _dat_n_samples(filename, dtype=None, n_channels=None, offset=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(filename) - offset) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def captured_logging(name=None):
    buffer = StringIO()
    logger = logging.getLogger(name)
    handlers = logger.handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield buffer
    buffer.flush()
    logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)


def _save_arrays(path, arrays):
    """Save multiple arrays in a single file by concatenating them along
    the first axis.
    A second array is stored with the offsets.
    """
    assert path.endswith('.npy')
    path = op.splitext(path)[0]
    offsets = np.cumsum([arr.shape[0] for arr in arrays])
    if not len(arrays):
        return
    concat = np.concatenate(arrays, axis=0)
    np.save(path + '.npy', concat)
    np.save(path + '.offsets.npy', offsets)


def _load_arrays(path):
    assert path.endswith('.npy')
    if not op.exists(path):
        return []
    path = op.splitext(path)[0]
    concat = np.load(path + '.npy')
    offsets = np.load(path + '.offsets.npy')
    return np.split(concat, offsets[:-1], axis=0)


def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.
    Avoid a copy if possible.
    """
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    return out


def _concatenate(arrs):
    if arrs is None:
        return
    arrs = [_as_array(arr) for arr in arrs if arr is not None]
    if not arrs:
        return
    return np.concatenate(arrs, axis=0)


def _ensure_dir_exists(path):
    if not op.exists(path):
        os.makedirs(path)


def _klusta_user_dir():
    """Return the absolute path to the klusta user directory."""
    home = op.expanduser("~")
    path = op.realpath(op.join(home, '.klusta'))
    return path


def _excerpt_step(n_samples, n_excerpts=None, excerpt_size=None):
    """Compute the step of an excerpt set as a function of the number
    of excerpts or their sizes."""
    assert n_excerpts >= 2
    step = max((n_samples - excerpt_size) // (n_excerpts - 1),
               excerpt_size)
    return step


def chunk_bounds(n_samples, chunk_size, overlap=0):
    """Return chunk bounds.
    Chunks have the form:
        [ overlap/2 | chunk_size-overlap | overlap/2 ]
        s_start   keep_start           keep_end     s_end
    Except for the first and last chunks which do not have a left/right
    overlap.
    This generator yields (s_start, s_end, keep_start, keep_end).
    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end - overlap // 2
    yield s_start, s_end, keep_start, keep_end

    while s_end - overlap + chunk_size < n_samples:
        s_start = s_end - overlap
        s_end = s_start + chunk_size
        keep_start = keep_end
        keep_end = s_end - overlap // 2
        if s_start < s_end:
            yield s_start, s_end, keep_start, keep_end

    s_start = s_end - overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    if s_start < s_end:
        yield s_start, s_end, keep_start, keep_end


def excerpts(n_samples, n_excerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    assert n_excerpts >= 2
    step = _excerpt_step(n_samples,
                         n_excerpts=n_excerpts,
                         excerpt_size=excerpt_size)
    for i in range(n_excerpts):
        start = i * step
        if start >= n_samples:
            break
        end = min(start + excerpt_size, n_samples)
        yield start, end


def data_chunk(data, chunk, with_overlap=False):
    """Get a data chunk."""
    assert isinstance(chunk, tuple)
    if len(chunk) == 2:
        i, j = chunk
    elif len(chunk) == 4:
        if with_overlap:
            i, j = chunk[:2]
        else:
            i, j = chunk[2:]
    else:
        raise ValueError("'chunk' should have 2 or 4 elements, "
                         "not {0:d}".format(len(chunk)))
    return data[i:j, ...]


def get_excerpts(data, n_excerpts=None, excerpt_size=None):
    assert n_excerpts is not None
    assert excerpt_size is not None
    if len(data) < n_excerpts * excerpt_size:
        return data
    elif n_excerpts == 0:
        return data[:0]
    elif n_excerpts == 1:
        return data[:excerpt_size]
    out = np.concatenate([data_chunk(data, chunk)
                          for chunk in excerpts(len(data),
                                                n_excerpts=n_excerpts,
                                                excerpt_size=excerpt_size)])
    assert len(out) <= n_excerpts * excerpt_size
    return out
