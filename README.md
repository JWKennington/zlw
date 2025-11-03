# zlw — Zero‑Latency Whitening utilities

zlw is a small, focused package that provides zero‑latency whitening utilities for gravitational‑wave data analysis. It includes:

- Whitening filter utilities
  - Minimum‑phase, zero‑latency whitening filters and helpers
  - Supporting Fourier/window helpers for stable, real‑time responses
- MP–MP scheme PSD drift correction terms
  - Utilities to compute first‑ and second‑order timing and phase correction terms
  - Tools to account for slow PSD mismatches between template and data

Where things live:
- zlw.kernels: minimum‑phase whitening filter construction and frequency‑response utilities
- zlw.fourier, zlw.window: helpers used by the whitening filters
- zlw.corrections: MP–MP scheme PSD drift correction terms (class MPMPCorrection)
- zlw/tests: basic tests (e.g., tests/test_kernels.py)
- zlw/src/zlw/bin: small simulation/QA scripts

Quick examples
- Whitening filter
  
  from zlw.kernels import MPWhiteningFilter
  
  # psd: one‑sided PSD array (Hz^-1), fs: sampling rate (Hz), n_fft: FFT length
  wf = MPWhiteningFilter(psd, fs, n_fft)
  Wf = wf.frequency_response()  # one‑sided frequency response (complex for min‑phase)

- MP–MP correction terms
  
  import numpy as np
  from zlw.corrections import MPMPCorrection
  
  # freqs: one‑sided frequency grid; psd1: data PSD; psd2: template PSD; htilde: template FFT
  corr = MPMPCorrection(freqs=freqs, psd1=psd1, psd2=psd2, htilde=htilde, fs=fs)
  # Simple first‑order corrections
  dt1, dphi1 = corr.simplified_correction()
  # Or the full second‑order set (includes cross terms)
  (dt1, dphi1), (dt2, dphi2) = corr.full_correction()

Notes
- “Zero‑latency” refers to the use of minimum‑phase whitening filters so that the whitening operation does not introduce group delay in the time domain.
- The MP–MP correction utilities follow the perturbative scheme that expands about the ratio of PSDs, providing drift terms for coalescence time and phase when the whitening filters differ slightly.

<!-- index.rst content start -->

<h1 align="center">SGN Documentation</h1>

<p align="center">
  <a href="https://git.ligo.org/greg/sgn/-/pipelines/latest">
    <img alt="ci" src="https://git.ligo.org/greg/sgn/badges/main/pipeline.svg" />
  </a>
  <a href="https://git.ligo.org/greg/sgn/-/pipelines/latest">
    <img alt="ci" src="https://git.ligo.org/greg/sgn/badges/main/coverage.svg" />
  </a>
  <a href="https://greg.docs.ligo.org/sgn/">
    <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat" />
  </a>
  <a href="https://pypi.org/project/sgn/">
    <img alt="pypi version" src="https://img.shields.io/pypi/v/sgn.svg" />
  </a>
</p>


SGN is a lightweight Python library for creating and executing task graphs
asynchronously for streaming data. With only builtin-dependencies, SGN is easy
to install and use. This page is for the base library `sgn`, but there is a
family of libraries that extend the functionality of SGN, including:

- [`sgn-ts`](https://git.ligo.org/greg/sgn-ts): TimeSeries utilities for SGN
- [`sgn-ligo`](https://git.ligo.org/greg/sgn-ligo): LSC specific utilities for SGN

## Installation

To install SGN, simply run:

```bash
pip install sgn
```

SGN has no dependencies outside of the Python standard library, so it should be
easy to install on any system.

## Quickstart

To get started with SGN, you can create a simple task graph that represents a
simple data processing pipeline with integers. Here's an example:

```python
import functools
from sgn import CallableTransform, CollectSink, IterSource, Pipeline


# Define a function to use in the pipeline
def scale(frame, factor: float):
    return None if frame.data is None else frame.data * factor


# Create source element
src = IterSource(
    name="src1",
    source_pad_names=["H1"],
    iters={"src1:src:H1": [1, 2, 3]},
)

# Create a transform element using an arbitrary function
trn1 = CallableTransform.from_callable(
    name="t1",
    sink_pad_names=["H1"],
    callable=functools.partial(scale, factor=10),
    output_pad_name="H1",
)

# Create the sink so we can access the data after running
snk = CollectSink(
    name="snk1",
    sink_pad_names=("H1",),
)

# Create the Pipeline
p = Pipeline()

# Connect elements using pipeline.connect()
p.connect(src, trn1)  # Connects matching pad names automatically
p.connect(trn1, snk)  # H1 -> H1

# Run the pipeline
p.run()

# Check the result of the sink queue to see outputs
assert list(snk.collects["snk1:snk:H1"]) == [10, 20, 30]

```

### Advanced Grouping and Selection

SGN also supports grouping elements and selecting specific pads for more complex pipelines:

```python
from sgn import Pipeline, IterSource, NullSink
from sgn.groups import group, select

# Create multiple sources
src1 = IterSource(name="src1", source_pad_names=["H1"])
src2 = IterSource(name="src2", source_pad_names=["L1", "V1"])

# Create sink
sink = NullSink(name="sink", sink_pad_names=["H1", "L1"])

# Group sources and select specific pads
sources = group(src1, select(src2, "L1"))  # Include all of src1, only L1 from src2

pipeline = Pipeline()
pipeline.connect(sources, sink)  # Automatic matching: H1->H1, L1->L1
```

The above example can be modified to use any data type, including json-friendly
nested dictionaries, lists, and strings. The `CallableTransform` class can be
used to create a transform element using any arbitrary function. The
`DequeSource` and `DequeSink` classes are used to create source and sink elements
that use `collections.deque` to store data.

## General Concepts

### Graph Construction

- **Sources**: Sources are the starting point of a task graph. They produce
  data that can be consumed by other tasks.

- **Transforms**: Transforms are tasks that consume data from one or more
  sources, process it, and produce new data.

- **Sinks**: Sinks are tasks that consume data from one or more sources and do
  something with it. This could be writing the data to a file, sending it over
  the network, or anything else.

### Control Flow

Using these concepts, you can create complex task graphs using SGN that process
and move data in a variety of ways. The SGN library provides a simple API for
creating and executing task graphs, with a few key types:

- **Frame**: A frame is a unit of data that is passed between tasks in a task
  graph. Frames can contain any type of data, and can be passed between tasks
  in a task graph.

- **Pad**: A pad is a connection point between two tasks in a task graph. Pads
  are used to pass frames between tasks, and can be used to connect tasks in a
  task graph. An edge is a connection between two pads in a task graph.

- **Element**: An element is a task in a task graph. Elements can be sources,
  transforms, or sinks, and can be connected together to create a task graph.

- **Pipeline**: A pipeline is a collection of elements that are connected
  together to form a task graph. Pipelines can be executed to process data, and
  can be used to create complex data processing workflows.

## Developer's Guide

SGN will execute a fixed graph of "pads", which are asynchronous function calls
bound to classes called "elements".

Data must have an origin and a end point in all graphs. These are called
sources and sinks.  Elements that create data are called source elements and
elements that collect data are called sink elements.  Likewise, pads on
elements are also called source and sink pads.  Data passed between pads are
stored in a Frame.

```
    /       ----------------------      <
   /       |   Source Element 1   |      \
  /        |                      |       \
 /          ---[source pad 'a']---         \
|                     |                     \
|                     | data flow            | The event loop runs this graph over and
\                     V                      | over pulling data through the pads
 \          ---[sink pad 'x'] ---           /
  \        |                     |         /
   \       |   Sink Element 1    |        /
    >      |                     |       /
           ---------------------       /
```

The whole graph execution is orchestrated by an event loop that will execute
until end of stream.  Here is a simple example implementing the above graph

```{.python notest}
from sgn.base import SourceElement, SinkElement, Frame
from sgn.apps import Pipeline

class MySourceClass(SourceElement):
    def new(self, pad):
        return Frame(data="hello")

class MySinkClass(SinkElement):
    def pull(self, pad, frame):
        print (frame.data)

source = MySourceClass(source_pad_names = ("a",))
sink = MySinkClass(sink_pad_names = ("x",))

pipeline = Pipeline()

pipeline.insert(source, sink, link_map = {sink.snks["x"]: source.srcs["a"]})

pipeline.run()
```

If you run this, it will run forever and you will see

```
hello
hello
hello
hello
hello
hello
hello
hello
hello
hello
...
```


You would need to send SIG INT or SIG kill to stop the program. Lets add a
feature to end the stream after 10 Frames.

```python
from sgn.base import SourceElement, SinkElement, Frame
from sgn.apps import Pipeline

class MySourceClass(SourceElement):
    def __post_init__(self):
        super().__post_init__()
        self.cnt = 0
    def new(self, pad):
        self.cnt += 1
        return Frame(data="hello", EOS=self.cnt > 10)

class MySinkClass(SinkElement):
    def pull(self, pad, frame):
        if frame.EOS:
            self.mark_eos(pad)
        print (frame.data)

source = MySourceClass(source_pad_names = ("a",))
sink = MySinkClass(sink_pad_names = ("x",))

pipeline = Pipeline()

pipeline.insert(source, sink, link_map = {sink.snks["x"]: source.srcs["a"]})

pipeline.run()
```

Now you would see the word "hello" printed 11 times.  The 11th time the Frame
is marked as EOS, which means end of stream.  The sink class checks the data it
has gotten and marks the pad as EOS.  When all sink element sink pads are at
EOS the pipeline stops running (in this case there is just one sink element
with one sink pad).

What if we want more than one pad?  It is possible to have many source and sink
pads on an element. SGN provides basic bookkeeping utilities for you, but
generally what the "correct" behavior is is up to you. Lets try a more
complicated example with multiple pads:

```
 ---------------------------------------------
|                                             |
|              Source Element 1               |
|                                             |
 --- [source pad 'a'] --- [source pad 'b'] ---
           |                 |
           | data flow       |
           V                 V
 --- [sink pad 'x'  ] --- [sink pad 'y'  ] ---
|                                             |
|               Sink Element 1                |
|                                             |
----------------------------------------------
```

```python
from dataclasses import dataclass
from sgn.base import SourceElement, SinkElement, Frame
from sgn.apps import Pipeline

@dataclass
class MySourceClass(SourceElement):
    # Of the form {"pad name": <data to put on the pad}
    pad_str_map: dict=None
    def __post_init__(self):
        # We will just use pad_str_map to define the source pad names too
        self.source_pad_names = tuple(self.pad_str_map)
        super().__post_init__()
        # save a pad map also hashed by pad not the string
        # NOTE: this must be done after super() post init so that the source pads exist
        self.pad_map = {self.srcs[p]: d for p,d in self.pad_str_map.items()}
        self.cnt = 0
    def new(self, pad):
        self.cnt += 1
        return Frame(data=self.pad_map[pad], EOS=self.cnt > 10)

class MySinkClass(SinkElement):
    def pull(self, pad, frame):
        if frame.EOS:
            self.mark_eos(pad)
        print (frame.data)

source = MySourceClass(pad_str_map = {"a": "Hello!", "b":"How are you?"})
sink = MySinkClass(sink_pad_names = ("x","y"))

pipeline = Pipeline()

pipeline.insert(source, sink, link_map = {sink.snks["x"]: source.srcs["a"], sink.snks["y"]: source.srcs["b"],})

pipeline.run()
```

Running this produces the following output:

```
e1-056827:~ crh184$ ./sgn-readme
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
```

Note that the total number of outputs is 12.  We had the counter in the new()
method which is a pad dependent method. It will be called once for each pad
during each loop iteration.  What if we wanted 10 loop iterations before
sending EOS? There is a convenient "internal" pad inside of every element that
is guaranteed to be called before any source pads and after any sink pads. Let's
modify the code to use that:

```python
from dataclasses import dataclass
from sgn.base import SourceElement, SinkElement, Frame
from sgn.apps import Pipeline

@dataclass
class MySourceClass(SourceElement):
    # Of the form {"pad name": <data to put on the pad}
    pad_str_map: dict=None
    def __post_init__(self):
        # We will just use pad_str_map to define the source pad names too
        self.source_pad_names = tuple(self.pad_str_map)
        super().__post_init__()
        # save a pad map also hashed by pad not the string
        # NOTE: this must be done after super() post init so that the source pads exist
        self.pad_map = {self.srcs[p]: d for p,d in self.pad_str_map.items()}
        self.cnt = 0
    def internal(self):
        self.cnt += 1
    def new(self, pad):
        return Frame(data=self.pad_map[pad], EOS=self.cnt > 10)

class MySinkClass(SinkElement):
    def pull(self, pad, frame):
        if frame.EOS:
            self.mark_eos(pad)
        print (frame.data)

source = MySourceClass(pad_str_map = {"a": "Hello!", "b":"How are you?"})
sink = MySinkClass(sink_pad_names = ("x","y"))

pipeline = Pipeline()

pipeline.insert(source, sink, link_map = {sink.snks["x"]: source.srcs["a"], sink.snks["y"]: source.srcs["b"],})

pipeline.run()
```

Now the output has the expected number of iterations
```
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
Hello!
How are you?
```

We can also use the internal method to make a more useful sink output, e.g.,

```python
from dataclasses import dataclass
from sgn.base import SourceElement, SinkElement, Frame
from sgn.apps import Pipeline

@dataclass
class MySourceClass(SourceElement):
    # Of the form {"pad name": <data to put on the pad}
    pad_str_map: dict=None
    def __post_init__(self):
        # We will just use pad_str_map to define the source pad names too
        self.source_pad_names = tuple(self.pad_str_map)
        super().__post_init__()
        # save a pad map also hashed by pad not the string
        # NOTE: this must be done after super() post init so that the source pads exist
        self.pad_map = {self.srcs[p]: d for p,d in self.pad_str_map.items()}
        self.cnt = 0
    def internal(self):
        self.cnt += 1
    def new(self, pad):
        return Frame(data=self.pad_map[pad], EOS=self.cnt > 10)

class MySinkClass(SinkElement):
    def __post_init__(self):
        super().__post_init__()
        self.combined_string = ""
    def internal(self):
        print (self.combined_string)
        self.combined_string = ""
    def pull(self, pad, frame):
        if frame.EOS:
            self.mark_eos(pad)
        self.combined_string += " %s" % frame.data

source = MySourceClass(pad_str_map = {"a": "Hello!", "b":"How are you?"})
sink = MySinkClass(sink_pad_names = ("x","y"))

pipeline = Pipeline()

pipeline.insert(source, sink, link_map = {sink.snks["x"]: source.srcs["a"], sink.snks["y"]: source.srcs["b"],})

pipeline.run()
```

which now produces

```
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
 Hello! How are you?
```

Graphs can have other elements called "transform elements." These have both
source and sink pads.  Also, it is possible to connect a source pad to multiple
sink pads (but not the other way around). Lets try to implement this graph

```
 ---------------------------------------------
|                                             |
|              Source Element 1               |
|                                             |
 --- [source pad 'a'] --- [source pad 'b'] ---
           |\                |\
           | \               | \
           |  \              |  \_________________________________________
           |   \_____________|_________________________                   \
           |                 |                         \                   \
           |                 |                          V                   V
           |                 |                 --- [sink pad 'l'  ] --- [sink pad 'm'  ] ---
           |                 |                |                                             |
           |                 |                |            Transform Element 1              |
           |                 |                |                                             |
           |                 |                 ------------- [source pad 'n'] --------------
           |                 |                                   /
           |                 |                                  /
           |                 |                                 /
           |                 |                                /
           |                 |                               /
           | data flow       |                              /
           V                 V                             V
 --- [sink pad 'x'  ] --- [sink pad 'y'  ] --- [sink pad 'z'  ] ---
|                                                                  |
|               Sink Element 1                                     |
|                                                                  |
-------------------------------------------------------------------
```

```python
from dataclasses import dataclass
from sgn.base import SourceElement, SinkElement, TransformElement, Frame
from sgn.apps import Pipeline

@dataclass
class MySourceClass(SourceElement):
    # Of the form {"pad name": <data to put on the pad}
    pad_str_map: dict=None
    def __post_init__(self):
        # We will just use pad_str_map to define the source pad names too
        self.source_pad_names = tuple(self.pad_str_map)
        super().__post_init__()
        # save a pad map also hashed by pad not the string
        # NOTE: this must be done after super() post init so that the source pads exist
        self.pad_map = {self.srcs[p]: d for p,d in self.pad_str_map.items()}
        self.cnt = 0
    def internal(self):
        self.cnt += 1
    def new(self, pad):
        return Frame(data=self.pad_map[pad], EOS=self.cnt > 10)

class MyTransformClass(TransformElement):
    def __post_init__(self):
        # written to assume a single source pad
        assert len(self.source_pad_names) == 1
        super().__post_init__()
        self.out_string = ""
        self.out_frame = None
        self.EOS = False
    def pull(self, pad, frame):
        self.out_string += " %s" % frame.data
        self.EOS |= frame.EOS
    def internal(self):
        # Reverse the data for fun.
        self.outframe = Frame(data=self.out_string[::-1], EOS=self.EOS)
        self.out_string = ""
    def new(self, pad):
        # This element just has one source pad
        return self.outframe


class MySinkClass(SinkElement):
    def __post_init__(self):
        super().__post_init__()
        self.combined_string = ""
    def internal(self):
        print (self.combined_string)
        self.combined_string = ""
    def pull(self, pad, frame):
        if frame.EOS:
            self.mark_eos(pad)
        self.combined_string += " %s" % frame.data

source = MySourceClass(pad_str_map = {"a": "Hello!", "b":"How are you?"})
transform = MyTransformClass(sink_pad_names = ("l","m",), source_pad_names = ("n",))
sink = MySinkClass(sink_pad_names = ("x","y","z"))

pipeline = Pipeline()

pipeline.insert(source,
               transform,
               sink,
               link_map = {sink.snks["x"]: source.srcs["a"],
                           sink.snks["y"]: source.srcs["b"],
                           transform.snks["l"]: source.srcs["a"],
                           transform.snks["m"]: source.srcs["b"],
                           sink.snks["z"]: transform.srcs["n"]
                          }
               )

pipeline.run()
```
which produces

```
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
 Hello! How are you? ?uoy era woH !olleH
```
### All you need to know about pads and names

Pads are hashable and they also have string names (though that name is not used
as the hash).  When developing you might get a bit turned around about how to
access and reference pads by name.  Here are a few rules:

- Elements have a notion of a short pad name.  These are verbatim what get
  passed to `source_pad_names` and `sink_pad_names`.
- The Element base classes will initialize pads with long pad names of the form
  `<element name>:["snk" | "source"]:<short name>`.
- These long names are almost never needed for anything programmatically but
  they can be handy to print out because they carry extra information encoded
  in the name.
- Usually you will use helper attributes to reference pads by their short names
  or to look up a pad's short name.

Below is a bit of interactive python code that should be all you need to sort
this out.

```{.python notest}
>>> from sgn.base import SourceElement
>>> e = SourceElement(name="example", source_pad_names=("alice","bob"))
>>> # Here are some relevant ways to access pad information
>>> # All of the "short" names -- these will be the strings provided by source_pad_names in the initialization
>>> print (e.source_pad_names)
('alice', 'bob')
>>> # A dictionary mapping the short name to a given pad object, e.g.,
>>> p = e.srcs["alice"]
>>> print (type(p))
<class 'sgn.base.SourcePad'>
>>> # The pad's long name
>>> print (p.name)
example:src:alice
>>> # A reverse dictionary mapping a pad to a short name
>>> print (e.rsrcs[p])
alice
```

### Some useful API docs from this guide
Below are some API docs for concepts that came up in this guide

- [SourceElement](https://greg.docs.ligo.org/sgn/sgn.base.html#sgn.base.SourceElement)
- [TransformElement](https://greg.docs.ligo.org/sgn/sgn.base.html#sgn.base.TransformElement)
- [SinkElement](https://greg.docs.ligo.org/sgn/sgn.base.html#sgn.base.SinkElement)
- [Frame](https://greg.docs.ligo.org/sgn/sgn.base.html#sgn.base.Frame)
- [Pipeline](https://greg.docs.ligo.org/sgn/sgn.apps.html#sgn.apps.Pipeline)
