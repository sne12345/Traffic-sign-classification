	|&??)|?@|&??)|?@!|&??)|?@	?︲D???︲D??!?︲D??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6|&??)|?@?+J	)y@1˅ʿ?a@AZ??/-???I?ZdK)@YT? Pō??*	?K7??'?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratoruX??@!Z?yn?bX@)uX??@1Z?yn?bX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?i??_=??!?????)??)?i??_=??1?????)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?4`??i??!?MuQp?@)?-?R???1?N{?5???:Preprocessing2F
Iterator::Model?Ēr?9??!G????E@)??QF\ z?1??ŋ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?@-#@!?????eX@){?\?&?k?1<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 71.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?︲D??I[??mْR@Q????w?9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?+J	)y@?+J	)y@!?+J	)y@      ??!       "	˅ʿ?a@˅ʿ?a@!˅ʿ?a@*      ??!       2	Z??/-???Z??/-???!Z??/-???:	?ZdK)@?ZdK)@!?ZdK)@B      ??!       J	T? Pō??T? Pō??!T? Pō??R      ??!       Z	T? Pō??T? Pō??!T? Pō??b      ??!       JGPUY?︲D??b q[??mْR@y????w?9@