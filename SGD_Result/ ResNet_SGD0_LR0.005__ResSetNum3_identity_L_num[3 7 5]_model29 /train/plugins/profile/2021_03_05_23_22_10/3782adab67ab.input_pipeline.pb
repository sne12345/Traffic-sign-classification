	??R".?@??R".?@!??R".?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??R".?@?gǬ?v@1R?G?+b@A?,??????I`???%@*	?"?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatoru=?u?7$@!??~?X@)u=?u?7$@1??~?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchus??=A??!w7ůY??)us??=A??1w7ůY??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??vN?@??!&ؕ1x??)?HZ????1b*?Bz??:Preprocessing2F
Iterator::Model }??AѸ?!???c?b??)yxρ?y?1ȉ?ݼ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMape??9$@!#?8?:?X@)??z2??k?1?"?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIEd?+?Q@Q?nBQ?<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?gǬ?v@?gǬ?v@!?gǬ?v@      ??!       "	R?G?+b@R?G?+b@!R?G?+b@*      ??!       2	?,???????,??????!?,??????:	`???%@`???%@!`???%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qEd?+?Q@y?nBQ?<@