from collections import namedtuple
import copy
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, SupportsIndex, Tuple, Union
from torch import Tensor, tensor
from torch_geometric.typing import SparseTensor
from torch_geometric.data.data import Data, BaseData, GlobalStorage, BaseStorage, size_repr
from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage

class LevelStorage(GlobalStorage):
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, _key: Optional[str] = None, **kwargs):
        if _key: self._level = int(_key.split('_')[-1])
        super().__init__(_mapping, _key=_key, **kwargs)

    @property
    def _key(self) -> str:
        return self.__dict__.get('_key', None)
    
    @_key.setter
    def _key(self, key: Union[int, str]):
        if isinstance(key, int):
            self._level = key
            self.__dict__['_key'] = f'level_{key}'
        elif isinstance(key, str):
            self.__dict__['_key'] = key
            self._level = int(key.split('_')[-1])
        else:
            raise TypeError()

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

class HData(BaseData):
    def __init__(self, x: Optional[List[Tensor]] = None, edge_index: Optional[List[Tensor]] = None, edge_attr: Optional[List[Tensor]] = None, y: Optional[List[Tensor]] = None, pos: Optional[List[Tensor]] = None, assignments: Optional[List[Tensor]] = None, **kwargs: List[Tensor]):
        super().__init__()
        self.__dict__['_global_store'] = BaseStorage(_parent=self)
        store_l = LevelStorage(_parent=self, _key='level_0')
        if x is not None:
            store_l['x'] = x.pop(0)
        if edge_index is not None:
            store_l['edge_index'] = edge_index.pop(0)
        if edge_attr is not None:
            store_l['edge_attr'] = edge_attr.pop(0)
        if y is not None:
            store_l['y'] = y.pop(0)
        if pos is not None:
            store_l['pos'] = pos.pop(0)
        if len(kwargs) > 0:
            for k in kwargs:
                store_l[k] = kwargs[k].pop(0)

        self.__dict__['_level_stores'] = list(store_l)
        self.add_levels(x, edge_index, edge_attr, pos, assignments, **kwargs)
        
    
    @staticmethod
    def from_single(data: Data):
        hdata = HData()
        
        if hasattr(data, "time_steps"): hdata.time_steps = data.time_steps
        if hasattr(data, "simid"): hdata.simid = data.simid
        del data.time_steps, data.simid
        hdata.__dict__['_level_stores'].append(LevelStorage(data.stores[0], _parent=hdata, _key='level_0'))

        return hdata
    
    def add_level(self, x: Optional[Tensor] = None, edge_index: Optional[Tensor] = None, edge_attr: Optional[Tensor] = None, pos: Optional[Tensor] = None, assignments: Optional[Tensor] = None, **kwargs):
        store_l = LevelStorage(_parent=self, _key=f'level_{len(self._level_stores)}')
        if x is not None:
            store_l['x'] = x
        if edge_index is not None:
            store_l['edge_index'] = edge_index
        if edge_attr is not None:
            store_l['edge_attr'] = edge_attr
        if pos is not None:
            store_l['pos'] = pos
        if assignments is not None:
            store_l['assignments'] = assignments
        if len(kwargs) > 0:
            store_l.update(kwargs)

        self._level_stores.append(store_l)

    def add_levels(self, x: Optional[List[Tensor]] = None, edge_index: Optional[List[Tensor]] = None, edge_attr: Optional[List[Tensor]] = None, pos: Optional[List[Tensor]] = None, assignments: Optional[List[Tensor]] = None, **kwargs):
        level_args = kwargs
        if x is not None:
            level_args['x'] = x
        if edge_index is not None:
            level_args['edge_index'] = edge_index
        if edge_attr is not None:
            level_args['edge_attr'] = edge_attr
        if pos is not None:
            level_args['pos'] = pos
        if assignments is not None:
            level_args['assignments'] = assignments
        
        for elem in zip(*level_args.items()):
            store_l = LevelStorage(_parent=self, _key=f'level_{len(self._level_stores)}')
            print(elem)
            for key, val in elem:
                store_l[key] = val

            self._level_stores.append(store_l)

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_global_store`.
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        out = [getattr(s, key) for s in self._level_stores if hasattr(s, key)]
        if len(out) > 0:
            return out
        elif hasattr(self._global_store, key):
            return getattr(self._global_store, key)
        raise AttributeError(f"'{self.__class__.__name__}' has no "
                             f"attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) == len(self._level_stores):
                for s, v in zip(self._level_stores, value):
                    setattr(s, key, v)
            else: 
                raise ValueError("Length of given value list doesn't fit to the number of levels in the dataset.")
        elif not hasattr(self, key) or hasattr(self._global_store, key):
            setattr(self._global_store, key, value)
        elif len(self._level_stores) == 1:
            setattr(self._level_stores[0], key, value)
        else:
            # Value or Attribute error
            raise ValueError("Key is used in hierarchy graph levels. Chose a level first to set a single value.\nE.g. `hdata[0].key = val`.")

    def __delattr__(self, key: str):
        for s in self.stores:
            delattr(s, key)

    def __getitem__(self, args: Union[Union[SupportsIndex, slice], str, Tuple[Union[SupportsIndex, slice], str]]) -> Any:
        # `data[*]` => Link to either `_global_store`, _node_store_dict` or
        # `_edge_store_dict`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.

        if isinstance(args, int):
            return self._level_stores[args] # TODO evtl. wie iter auf Data umstellen
        elif isinstance(args, str):
            return getattr(self, args)
        elif isinstance(args, (SupportsIndex, slice)):
            self._level_stores[args]
        elif isinstance(args, tuple):
            level, key = (args[1], args[0]) if isinstance(args[0], str) else args
            if key == 'level_':
                return self._level_stores[level]
            else:
                return getattr(self._level_stores[level], key)

    def __setitem__(self, key: Union[Tuple[Union[SupportsIndex, slice], str], str], value: Any):
        if isinstance(key, str):
            setattr(self, key, value)
        elif isinstance(key, tuple):
            level, attr = key[1], key[0] if isinstance(key[0], str) else key
            setattr(self._level_stores[level], attr, value)
        else:
            raise KeyError(f"Keys of type {type(key)} are not supported.")


    def __delitem__(self, key: Union[Tuple[Union[SupportsIndex, slice], str], str, Union[SupportsIndex, slice]]):
        # `del data[*]` => Link to `_node_store_dict` or `_edge_store_dict`.
        if isinstance(key, str):
            delattr(self, key)
        elif isinstance(key, (SupportsIndex, slice)):
            del self._level_stores[key]
        elif isinstance(key, tuple):
            level, attr = key[1], key[0] if isinstance(key[0], str) else key
            delattr(self._level_stores[level], attr)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_global_store'] = copy.copy(self._global_store)
        out._global_store._parent = out
        out.__dict__['_level_stores'] = [None] * len(self._level_stores)
        for l, s in enumerate(self._level_stores):
            os = copy.copy(s)
            os._parent = out
            out._level_stores[l] = os
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        for store in out._level_stores:
            store._parent = out
        return out

    def __repr__(self) -> str:
        info = [size_repr(f'level_{k}', v, 2) for k, v in enumerate(self._level_stores)]
        info = ',\n'.join(info)
        info = f'\n{info}\n' if len(info) > 0 else info
        return f'{self.__class__.__name__}({info})'

    def stores_as(self, data: 'HData'):
        ldiff = len(data.__dict__['_level_stores']) - len(self._level_stores)
        if ldiff > 0:
            for _ in range(ldiff):
                self.add_level()
        elif ldiff < 0:
            del self._level_stores[len(data.__dict__['_level_stores']):]
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._global_store] + self._level_stores

    @property
    def node_stores(self) -> List[NodeStorage]:
        return self._level_stores

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return self._level_stores

    def to_dict(self) -> Dict[str, Any]:
        out = {'global': self._global_store.to_dict()}
        for i, store in enumerate(self._level_stores):
            # out[f'level_{i}'] = store.to_dict()
            out[i] = store.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        field_names = [f'level_{i}' for i in range(len(self._level_stores))]
        field_values = [store.to_namedtuple() for store in self._level_stores]
        DataTuple = namedtuple('DataTuple', field_names)
        return DataTuple(*field_values)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key or key == 'assignments':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, store: LevelStorage, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return store.num_nodes
        elif key == 'assignments':
            return tensor(((self._level_stores[store._level - 1].num_nodes,), (store.num_nodes,)))
        else:
            return 0

    def __len__(self) -> int:
        return len(self._level_stores)
    
    @property
    def num_levels(self) -> int:
        return len(self._level_stores)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per particle in the graph."""
        nnf = self._level_stores[0].num_node_features
        if hasattr(self._level_stores[0], 'node_attr'):
            nnf += self[0]['node_attr'].size(-1)
        return nnf

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per particle in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the particle level graph."""
        nef = self._level_stores[0].num_edge_features
        if hasattr(self._level_stores[0], 'additional_message_features'):
            nef += self[0]['additional_message_features'].size(-1)
        return nef

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values."""
        for store_l in self._level_stores:
            level = Data()
            level.__dict__['_store'] = store_l # copy? -> keine Änderungen an attr mögl.
            yield level

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level attribute."""
        for store in self._level_stores:
            if hasattr(store, key):
                return store.is_node_attr(key)
        
        raise AttributeError(f"No attribute {key} found.")

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level attribute."""
        for store in self._level_stores:
            if hasattr(store, key):
                return store.is_edge_attr(key)
        
        raise AttributeError(f"No attribute {key} found.")

    @property
    def y(self) -> Any:
        return self._level_stores[0]['y'] if 'y' in self._level_stores[0] else None
