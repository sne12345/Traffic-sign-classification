	?{b?*#?@?{b?*#?@!?{b?*#?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?{b?*#?@????Nv@1?q7??o@A?#bJ$???Ip@KW??@*	?&1,ǵ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ڦx?@!3x%???X@)??ڦx?@13x%???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchV?6?㢲?!}2>?]???)V?6?㢲?1}2>?]???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1[?*¹?!U
F?U???)?l??}??1`_?????:Preprocessing2F
Iterator::Model???%VF??!? r}???)s?4?Bx?1j??x2??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??????@!|?7
??X@)?t?? ?k?1?H??h???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??w?C?M@QCS?W?BD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Nv@????Nv@!????Nv@      ??!       "	?q7??o@?q7??o@!?q7??o@*      ??!       2	?#bJ$????#bJ$???!?#bJ$???:	p@KW??@p@KW??@!p@KW??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??w?C?M@yCS?W?BD@