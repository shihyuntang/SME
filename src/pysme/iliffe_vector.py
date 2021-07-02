import io
import logging
import numpy as np

from flex.extensions.bindata import MultipleDataExtension

from .persistence import IPersist

logger = logging.getLogger(__name__)


class Iliffe_vector(MultipleDataExtension):
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, nseg=None, values=None, index=None, dtype=float):
        # sizes = size of the individual parts
        # the indices are then [0, s1, s1+s2, s1+s2+s3, ...]
        super().__init__()
        if values is not None and index is None:
            if isinstance(values, np.ndarray):
                values = [values]
            self._values = values
        elif values is not None and index is not None:
            if index[0] != 0:
                index = [0, *index]

            self._values = [
                values[index[i] : index[i + 1]] for i in range(len(index) - 1)
            ]

        elif nseg is not None:
            if values is not None:
                self._values = list(values[:nseg])
            else:
                self._values = [np.array([]) for _ in range(nseg)]
        else:
            self._values = []

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        if not hasattr(index, "__len__"):
            index = (index,)

        if len(index) == 0:
            return self._values

        if isinstance(index, range):
            index = list(index)

        if isinstance(index, (list, np.ndarray)):
            values = [self[i] for i in index]
            return Iliffe_vector(values=values)

        if isinstance(index, str):
            # This happens for example for np.recarrays
            return Iliffe_vector(values=self._values__[index])

        if isinstance(index, Iliffe_vector):
            if not self.__equal_size__(index):
                raise ValueError("Index vector has a different shape")
            values = [v[i] for v, i in zip(self._values, index._values)]
            return Iliffe_vector(values=values)

        if isinstance(index[0], slice):
            start = index[0].start if index[0].start is not None else 0
            stop = index[0].stop if index[0].stop is not None else len(self)
            step = index[0].step if index[0].step is not None else 1

            if stop > len(self):
                stop = len(self)

            values = self._values[start:stop]
            if len(index) > 1:
                if isinstance(index[1], (int, np.integer)):
                    values = [v[index[1]] for v in values]
                    return np.array(values)
                elif isinstance(index[1], (list, np.ndarray)):
                    if len(index[1]) == len(self):
                        values = [v[i] for v, i in zip(values, index[1])]
                        return np.array(values)
                values = [np.atleast_1d(v[index[1:]]) for v in values]
            return Iliffe_vector(values=values)

        if len(index) == 1:
            return self._values[index[0]]
        if len(index) == 2:
            return self._values[index[0]][index[1]]
        raise KeyError("Key must be maximum 2D")

    def __setitem__(self, index, value):
        if not hasattr(index, "__len__"):
            index = (index,)

        if isinstance(index, str):
            self._values[index] = value
            return

        if isinstance(index, Iliffe_vector):
            if not self.__equal_size__(index):
                raise ValueError("Index vector has a different shape")
            for i, ind in enumerate(index):
                self._values[i][ind] = value
            return

        if len(index) == 0:
            self._values = value
        elif len(index) == 1:
            tmp = self._values[index[0]]
            if isinstance(tmp, list):
                for t in tmp:
                    t[:] = value
            else:
                if np.isscalar(value):
                    tmp[:] = value
                else:
                    self._values[index[0]] = value
        elif len(index) == 2:
            self._values[index[0]][index[1]] = value
        else:
            raise KeyError("Key must be maximum 2D")

    # Math operators
    # If both are Iliffe vectors of the same size, use element wise operations
    # Otherwise apply the operator to _values
    def __equal_size__(self, other):
        if not isinstance(other, Iliffe_vector):
            return NotImplemented

        if self.shape[0] != other.shape[0]:
            return False

        return all(self.shape[1] == other.shape[1])

    def __operator__(self, other, operator):
        if isinstance(other, np.ndarray):
            # Proper broadcasting shape
            if other.shape[0] == len(self) and (
                other.ndim == 1 or (other.ndim == 2 and other.shape[1] == 1)
            ):
                values = [getattr(v, operator)(o) for v, o in zip(self._values, other)]
            else:
                raise ValueError(
                    f"Incompatible shapes of ({len(self)}) and {other.shape}"
                )
        elif isinstance(other, Iliffe_vector):
            if not self.__equal_size__(other):
                return NotImplemented
            other = other._values
            values = [getattr(v, operator)(o) for v, o in zip(self._values, other)]
        else:
            values = [getattr(v, operator)(other) for v in self._values]
        iv = Iliffe_vector(values=values)
        return iv

    def __eq__(self, other):
        return self.__operator__(other, "__eq__")

    def __ne__(self, other):
        return self.__operator__(other, "__ne__")

    def __lt__(self, other):
        return self.__operator__(other, "__lt__")

    def __gt__(self, other):
        return self.__operator__(other, "__gt__")

    def __le__(self, other):
        return self.__operator__(other, "__le__")

    def __ge__(self, other):
        return self.__operator__(other, "__ge__")

    def __add__(self, other):
        return self.__operator__(other, "__add__")

    def __sub__(self, other):
        return self.__operator__(other, "__sub__")

    def __mul__(self, other):
        return self.__operator__(other, "__mul__")

    def __truediv__(self, other):
        return self.__operator__(other, "__truediv__")

    def __floordiv__(self, other):
        return self.__operator__(other, "__floordiv__")

    def __mod__(self, other):
        return self.__operator__(other, "__mod__")

    def __divmod__(self, other):
        return self.__operator__(other, "__divmod__")

    def __pow__(self, other):
        return self.__operator__(other, "__pow__")

    def __lshift__(self, other):
        return self.__operator__(other, "__lshift__")

    def __rshift__(self, other):
        return self.__operator__(other, "__rshift__")

    def __and__(self, other):
        return self.__operator__(other, "__and__")

    def __or__(self, other):
        return self.__operator__(other, "__or__")

    def __xor__(self, other):
        return self.__operator__(other, "__xor__")

    def __radd__(self, other):
        return self.__operator__(other, "__radd__")

    def __rsub__(self, other):
        return self.__operator__(other, "__rsub__")

    def __rmul__(self, other):
        return self.__operator__(other, "__rmul__")

    def __rtruediv__(self, other):
        return self.__operator__(other, "__rtruediv__")

    def __rfloordiv__(self, other):
        return self.__operator__(other, "__rfloordiv__")

    def __rmod__(self, other):
        return self.__operator__(other, "__rmod__")

    def __rdivmod__(self, other):
        return self.__operator__(other, "__rdivmod__")

    def __rpow__(self, other):
        return self.__operator__(other, "__rpow__")

    def __rlshift__(self, other):
        return self.__operator__(other, "__rlshift__")

    def __rrshift__(self, other):
        return self.__operator__(other, "__rrshift__")

    def __rand__(self, other):
        return self.__operator__(other, "__rand__")

    def __ror__(self, other):
        return self.__operator__(other, "__ror__")

    def __rxor__(self, other):
        return self.__operator__(other, "__rxor__")

    def __iadd__(self, other):
        return self.__operator__(other, "__iadd__")

    def __isub__(self, other):
        return self.__operator__(other, "__isub__")

    def __imul__(self, other):
        return self.__operator__(other, "__imul__")

    def __itruediv__(self, other):
        return self.__operator__(other, "__itruediv__")

    def __ifloordiv__(self, other):
        return self.__operator__(other, "__ifloordiv__")

    def __imod__(self, other):
        return self.__operator__(other, "__imod__")

    def __ipow__(self, other):
        return self.__operator__(other, "__ipow__")

    def __iand__(self, other):
        return self.__operator__(other, "__iand__")

    def __ior__(self, other):
        return self.__operator__(other, "__ior__")

    def __ixor__(self, other):
        return self.__operator__(other, "__ixor__")

    def __neg__(self):
        values = -self._values
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __pos__(self):
        return self

    def __abs__(self):
        values = abs(self._values)
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __invert__(self):
        values = ~self._values
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __str__(self):
        s = [str(i) for i in self]
        s = str(s).replace("'", "")
        return s

    def __repr__(self):
        return f"Iliffe_vector({self.sizes}, {self._values})"

    def max(self):
        """ Maximum value in all segments """
        return np.max(self._values)

    def min(self):
        """ Minimum value in all segments """
        return np.min(self._values)

    def astype(self, dtype):
        for i, seg in enumerate(self):
            self._values[i] = seg.astype(dtype)
        return self

    @property
    def size(self):
        """int: number of elements in vector """
        return sum(self.sizes)

    @property
    def shape(self):
        """tuple(int, list(int)): number of segments, array with size of each segment """
        return len(self), self.sizes

    @property
    def sizes(self):
        """list(int): Sizes of the different segments """
        return np.asarray([len(v) for v in self._values])

    @property
    def ndim(self):
        """int: its always 2D """
        return 2

    @property
    def dtype(self):
        """dtype: numpy datatype of the values """
        return self._values[0].dtype

    @property
    def flat(self):
        """iter: Flat iterator through the values """
        for v in self._values:
            for i in v:
                yield i

    def flatten(self):
        """
        Returns a new(!) flattened version of the vector
        Values are identical to _values if the size
        of all segements equals the size of _values

        Returns
        -------
        flatten: array
            new flat (1d) array of the values within this Iliffe vector
        """
        return np.concatenate(self._values)

    def ravel(self):
        """
        View of the contained values as a 1D array.
        Not a copy

        Returns
        -------
        raveled: array
            1d array of the contained values
        """
        # TODO somehow avaoid making a copy?
        return self.flatten()

    def copy(self):
        """
        Create a copy of the current vector

        Returns
        -------
        copy : Iliffe_vector
            A copy of this vector
        """
        values = [np.copy(v) for v in self._values]
        return Iliffe_vector(values=values)

    def append(self, other):
        """
        Append a new segment to the end of the vector
        This creates new memory arrays for the values and the index
        """
        self._values.append(other)

    def _prepare(self, name: str):
        cls = self.__class__

        header_fname = f"{name}/header.json"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        result = [(header_info, header_bio)]

        for key, value in enumerate(self._values):
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
        for i, v in enumerate(self._values):
            obj[str(i)] = cls._np_to_dict(v)
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        data = {name: cls._np_from_dict(d) for name, d in data.items()}
        data = [data[str(i)] for i in range(len(data))]
        obj = cls(values=data)
        return obj

    def _save(self):
        data = {str(i): v for i, v in enumerate(self._values)}
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
        np.savez(b, *self._values)
        file.writestr(f"{folder}.npz", b.getvalue())

    @staticmethod
    def _load_v1(file):
        # file: npzfile
        names = file.files
        values = [file[n] for n in names]
        return Iliffe_vector(values=values)
