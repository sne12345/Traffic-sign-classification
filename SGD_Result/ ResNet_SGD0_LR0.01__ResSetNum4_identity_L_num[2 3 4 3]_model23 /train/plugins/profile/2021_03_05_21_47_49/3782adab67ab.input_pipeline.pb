	C9ѮZ?@C9ѮZ?@!C9ѮZ?@	7?y?nˣ?7?y?nˣ?!7?y?nˣ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6C9ѮZ?@??o?(?t@1J??І&g@A1??c???I????s?"@Y)%?????*	??C??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????(@!??ֿ?X@)?????(@1??ֿ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??6?[??!?EK?????)??6?[??1?EK?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????U-??!ru㼻m??)[A?+???1:?{?????:Preprocessing2F
Iterator::ModelzQ?_???!?{Z?,??)ެ???|?1<?ٹ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?l#?,@!d?bM?X@)?TPQ?+m?1ʝ6tdl??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no97?y?nˣ?IT???$P@Q?.?ϱA@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??o?(?t@??o?(?t@!??o?(?t@      ??!       "	J??І&g@J??І&g@!J??І&g@*      ??!       2	1??c???1??c???!1??c???:	????s?"@????s?"@!????s?"@B      ??!       J	)%?????)%?????!)%?????R      ??!       Z	)%?????)%?????!)%?????b      ??!       JGPUY7?y?nˣ?b qT???$P@y?.?ϱA@