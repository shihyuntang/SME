import io
import logging
import numpy as np
from numbers import Integral

import numpy.lib.mixins
from flex.flex import FlexExtension
from flex.extensions.bindata import MultipleDataExtension

logger = logging.getLogger(__name__)

# TODO make this a proper subclass of np.ndarray
# see also https://numpy.org/devdocs/user/basics.subclassing.html
HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class Iliffe_vector(numpy.lib.mixins.NDArrayOperatorsMixin, MultipleDataExtension):
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, values, dtype=None):
        FlexExtension.__init__(self, cls=MultipleDataExtension)

        sizes = [len(v) for v in values]
        offsets = np.array([0, *np.cumsum(sizes)])

        data = np.concatenate(values)
        if dtype is not None:
            data = data.astype(dtype)

        self.data = data
        self.offsets = offsets

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return self.__getsegment__(key)
        if isinstance(key, slice):
            key = range(self.nseg)[key]
            values = [self.__getsegment__(k) for k in key]
            return self.__class__(values)
        if isinstance(key, list):
            values = [self.__getsegment__(k) for k in key]
            return self.__class__(values)
        if isinstance(key, tuple):
            if isinstance(key[0], Integral):
                return self[key[0]][key[1]]
            if isinstance(key[0], list):
                values = [self.__getsegment__(k) for k in key[0]]
                values = [v[key[1]] for v in values]
                if isinstance(key[1], Integral):
                    return np.array(values)
                if isinstance(key[1], slice):
                    return self.__class__(values)
            if isinstance(key[0], slice):
                key0 = range(self.nseg)[key[0]]
                values = [self.__getsegment__(k) for k in key0]
                values = [v[key[1]] for v in values]
                if isinstance(key[1], Integral):
                    return np.array(values)
                if isinstance(key[1], slice):
                    return self.__class__(values)
        if isinstance(key, Iliffe_vector):
            return self.data[key.data]
        raise KeyError

    def __setitem__(self, key, value):
        isscalar = np.isscalar(value)
        if not isscalar:
            value = np.asarray(value)
        if isinstance(key, Integral):
            return self.__setsegment__(key, value)
        if isinstance(key, slice):
            key = range(self.nseg)[key]
            if isscalar or (value.ndim == 1 and value.dtype != "O"):
                for k in key:
                    self.__setsegment__(k, value)
            else:
                for i, k in enumerate(key):
                    self.__setsegment__(k, value[i])
            return
        if isinstance(key, tuple):
            if isinstance(key[0], Integral):
                data = self.__getsegment__(key[0])
                data[key[1]] = value
                return
            if isinstance(key[0], slice):
                key0 = range(self.nseg)[key[0]]
                if isscalar or (value.ndim == 1 and value.dtype != "O"):
                    for k in key0:
                        data = self.__getsegment__(k)
                        data[key[1]] = value
                else:
                    for i, k in enumerate(key0):
                        data = self.__getsegment__(k)
                        data[key[1]] = value
                return
        if isinstance(key, Iliffe_vector):
            self.data[key.data] = value
            return
        raise KeyError

    def __getsegment__(self, seg):
        while seg < 0:
            seg = self.nseg - seg
        if seg > self.nseg - 1:
            raise IndexError
        low, upp = self.offsets[seg : seg + 2]
        return self.data[low:upp]

    def __setsegment__(self, seg, value):
        while seg < 0:
            seg = self.nseg - seg
        if seg > self.nseg - 1:
            raise IndexError
        low, upp = self.offsets[seg : seg + 2]
        self.data[low:upp] = value

    def __array__(self, dtype=None):
        arr = self.data
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            inputs = list(inputs)
            for i, input in enumerate(inputs):
                if isinstance(input, self.__class__):
                    inputs[i] = input.__array__()
            if "out" in kwargs:
                out = []
                for i, o in enumerate(kwargs["out"]):
                    if isinstance(o, self.__class__):
                        out.append(o.__array__())
                    else:
                        out.append(o)
                kwargs["out"] = tuple(out)
            arr = ufunc(*inputs, **kwargs)
            arr = [arr[l:u] for l, u in zip(self.offsets[:-1], self.offsets[1:])]
            return self.__class__(arr)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.segments})"

    def __len__(self):
        return self.nseg

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (self.nseg, self.sizes)

    @property
    def size(self):
        return self.data.size

    @property
    def sizes(self):
        return np.diff(self.offsets)

    @property
    def nseg(self):
        return len(self.offsets) - 1

    @property
    def segments(self):
        return [self.data[l:u] for l, u in zip(self.offsets[:-1], self.offsets[1:])]

    @implements(np.ravel)
    def ravel(self):
        return self.data

    @implements(np.copy)
    def copy(self):
        data = self.segments
        data = [np.copy(seg) for seg in data]
        return self.__class__(data)

    @classmethod
    def from_indices(cls, array, indices):
        if indices is None:
            return cls([array])
        else:
            offsets = [0, *indices]
            arr = [array[l:u] for l, u in zip(offsets[:-1], offsets[1:])]
            return cls(arr)

    # For IO with Flex
    def _prepare(self, name: str):
        cls = self.__class__

        header_fname = f"{name}/header.json"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        result = [(header_info, header_bio)]

        for key, value in enumerate(self.segments):
            data_fname = f"{name}/{key}.npy"
            data_info, data_bio = cls._prepare_npy(data_fname, value)
            result += [(data_info, data_bio)]

        return result

    @classmethod
    def _parse(cls, header: dict, members: dict):
        data = {key[:-4]: cls._parse_npy(bio) for key, bio in members.items()}
        data = [data[str(i)] for i in range(len(data))]
        ext = cls(values=data)
        return ext

    def to_dict(self):
        cls = self.__class__
        obj = {"header": self.header}
        for i, v in enumerate(self.segments):
            obj[str(i)] = cls._np_to_dict(v)
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        data = {name: cls._np_from_dict(d) for name, d in data.items()}
        data = [data[str(i)] for i in range(len(data))]
        obj = cls(values=data)
        return obj

    def _save(self):
        data = {str(i): v for i, v in enumerate(self.segments)}
        ext = MultipleDataExtension(data=data)
        return ext

    @classmethod
    def _load(cls, ext: MultipleDataExtension):
        data = ext.data
        values = [data[str(i)] for i in range(len(data))]
        iv = cls(values=values)
        return iv

    def _save_v1(self, file, folder=""):
        """
        Creates a npz structure, representing the vector

        Returns
        -------
        data : bytes
            data to use
        """
        b = io.BytesIO()
        np.savez(b, *self.segments)
        file.writestr(f"{folder}.npz", b.getvalue())

    @classmethod
    def _load_v1(file):
        # file: npzfile
        names = file.files
        values = [file[n] for n in names]
        return cls(values=values)
