	???;?n[@???;?n[@!???;?n[@	??ƧRM????ƧRM??!??ƧRM??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???;?n[@??`;@1-??2?W@Aɐc???I?^? @Yr???_??*	??Mb?z?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?~?^??!@!????^?X@)?~?^??!@1????^?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?Q?=???!?;QȰ???)?Q?=???1?;QȰ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?8?j?3??!P???????)????il??1ʸZ????:Preprocessing2F
Iterator::Model(֩?=#??!j`("υ??)Rԙ{H?~?1?0?ҩ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?:????!@!???a??X@)???E_Az?1>??U??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??ƧRM??Ih?6?
?)@Q6eN?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??`;@??`;@!??`;@      ??!       "	-??2?W@-??2?W@!-??2?W@*      ??!       2	ɐc???ɐc???!ɐc???:	?^? @?^? @!?^? @B      ??!       J	r???_??r???_??!r???_??R      ??!       Z	r???_??r???_??!r???_??b      ??!       JGPUY??ƧRM??b qh?6?
?)@y6eN?U@