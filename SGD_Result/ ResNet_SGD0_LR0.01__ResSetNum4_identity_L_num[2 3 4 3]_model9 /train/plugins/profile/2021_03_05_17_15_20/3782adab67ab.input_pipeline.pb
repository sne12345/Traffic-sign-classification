	???N{@???N{@!???N{@	$??f*??$??f*??!$??f*??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???N{@?h?hsLs@1K %vm!\@A?đ??Iظ?]?y.@YK?46??*`??"?I?@)      ?=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??g??(@!D????X@)??g??(@1D????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??d?`T??!>????l??)??d?`T??1>????l??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?jׄ?Ƹ?!Jr]l????)???iOə?1*tV?????:Preprocessing2F
Iterator::Model
i?A'???!?a?a???)C?_?+?{?1h?D?(???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap]2????(@!?<	<??X@)???g?n?1??ӱ7???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9%??f*??I?4I^?R@Qo<6??9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?h?hsLs@?h?hsLs@!?h?hsLs@      ??!       "	K %vm!\@K %vm!\@!K %vm!\@*      ??!       2	?đ???đ??!?đ??:	ظ?]?y.@ظ?]?y.@!ظ?]?y.@B      ??!       J	K?46??K?46??!K?46??R      ??!       Z	K?46??K?46??!K?46??b      ??!       JGPUY%??f*??b q?4I^?R@yo<6??9@