	?f???{@?f???{@!?f???{@	? ??56??? ??56??!? ??56??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?f???{@?W???s@1???!Ɩ^@A???|@???I@??>?&@Y:???u??*	????ͺ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator? d??!@!?E#h?X@)? d??!@1?E#h?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???C?r??!	????)???C?r??1	????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn??Sr??!]A+????)???~???1avf=???:Preprocessing2F
Iterator::Model%?/????!ƶ??U??)O=?බ??1?VgM???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? ?S??!@!%?騮X@)???O??m?1?}????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9? ??56??If?A??"R@QH??Q[b;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?W???s@?W???s@!?W???s@      ??!       "	???!Ɩ^@???!Ɩ^@!???!Ɩ^@*      ??!       2	???|@??????|@???!???|@???:	@??>?&@@??>?&@!@??>?&@B      ??!       J	:???u??:???u??!:???u??R      ??!       Z	:???u??:???u??!:???u??b      ??!       JGPUY? ??56??b qf?A??"R@yH??Q[b;@