	?f??Mg@?f??Mg@!?f??Mg@	E<?g??E<?g??!E<?g??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?f??Mg@????$@1bN?&??d@A?u?!H??I??[<?g!@Y?#?&???*	i??|?N?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????6@!?N???X@)?????6@1?N???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?~k'J??!???0??)?~k'J??1???0??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismni5$?!:H"????)LU?????1???:
??:Preprocessing2F
Iterator::Model?}8gD??!??~????)DQ?O?I??1?{n????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap}%???6@!m`??X@)???
?b??11? ?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9E<?g??I`58??$@QG????VV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????$@????$@!????$@      ??!       "	bN?&??d@bN?&??d@!bN?&??d@*      ??!       2	?u?!H???u?!H??!?u?!H??:	??[<?g!@??[<?g!@!??[<?g!@B      ??!       J	?#?&????#?&???!?#?&???R      ??!       Z	?#?&????#?&???!?#?&???b      ??!       JGPUYE<?g??b q`58??$@yG????VV@