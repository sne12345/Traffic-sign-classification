	?E?~O?@?E?~O?@!?E?~O?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?E?~O?@Cs?FZ?u@1m?%?d@A?L?:???Iu!V?)&@*	I+g??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?	M##@!z?
h??X@)?	M##@1z?
h??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??ڧ?1??!??S????)??ڧ?1??1??S????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????ɽ?!?E?$4??)?{?ʄ_??1Y=n6? ??:Preprocessing2F
Iterator::Model?M?g\??!=?}gx7??)?4?($y?1҆S;5??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMaptzލ%#@!b"?X@)p\?M4o?1,
ջ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI:~???SQ@QN5ȱ>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Cs?FZ?u@Cs?FZ?u@!Cs?FZ?u@      ??!       "	m?%?d@m?%?d@!m?%?d@*      ??!       2	?L?:????L?:???!?L?:???:	u!V?)&@u!V?)&@!u!V?)&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q:~???SQ@yN5ȱ>@