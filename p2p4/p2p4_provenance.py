"""Owner-scoped CUDA capture provenance; raw handles are never lifetime keys."""
import weakref
from dataclasses import dataclass, field

@dataclass
class Capture:
    generation: int
    identity: tuple
    rows: list = field(default_factory=list)
    bindings: dict = field(default_factory=dict)
    sealed: bool = False

class Registry:
    def __init__(self):
        self.owners=weakref.WeakKeyDictionary()
        self.active={}
        self.generation=0
    def __len__(self):return len(self.owners)
    def begin(self,owner,identity):
        if identity in self.active:raise RuntimeError('Capture identity already active')
        self.invalidate(owner)
        self.generation+=1
        self.owners[owner]=Capture(self.generation,identity)
        self.active[identity]=weakref.ref(owner)
    def invalidate(self,owner):
        old=self.owners.pop(owner,None)
        if old is not None:self.active.pop(old.identity,None)
    def append(self,identity,transport,row):
        ref=self.active.get(identity);owner=None if ref is None else ref()
        if owner is None:raise RuntimeError('Collective capture has no live owner')
        record=self.owners[owner]
        if record.sealed:raise RuntimeError('Cannot append to sealed capture')
        transport.require()
        binding=transport.binding()
        previous=record.bindings.setdefault(id(transport),(transport,binding))
        if previous!=(transport,binding):raise RuntimeError('Capture communicator changed')
        record.rows.append(tuple(row))
    def seal(self,owner):
        record=self.owners[owner]
        if record.sealed:raise RuntimeError('Capture already sealed')
        self.active.pop(record.identity,None)
        record.sealed=True
    def get(self,owner):
        record=self.owners.get(owner)
        if record is None or not record.sealed:raise RuntimeError('Missing completed owner capture provenance')
        if owner.raw_cuda_graph()!=record.identity[1]:raise RuntimeError('Captured raw graph changed')
        for transport,binding in record.bindings.values():
            transport.require()
            if transport.binding()!=binding:raise RuntimeError('Captured communicator binding changed')
        return record

def install_hooks(graph_class,identity):
    """Hook the Python CUDAGraph lifecycle used by both vLLM and private banks."""
    if getattr(graph_class,'_dsv41_provenance_hooks',False):return
    begin,end,reset=graph_class.capture_begin,graph_class.capture_end,graph_class.reset
    def capture_begin(owner,*args,**kwargs):
        REGISTRY.invalidate(owner)
        result=begin(owner,*args,**kwargs)
        try:REGISTRY.begin(owner,identity())
        except BaseException:
            REGISTRY.invalidate(owner);raise
        return result
    def capture_end(owner,*args,**kwargs):
        try:result=end(owner,*args,**kwargs)
        except BaseException:
            REGISTRY.invalidate(owner);raise
        REGISTRY.seal(owner)
        return result
    def graph_reset(owner,*args,**kwargs):
        REGISTRY.invalidate(owner)
        return reset(owner,*args,**kwargs)
    graph_class.capture_begin=capture_begin
    graph_class.capture_end=capture_end
    graph_class.reset=graph_reset
    graph_class._dsv41_provenance_hooks=True

REGISTRY=Registry()
