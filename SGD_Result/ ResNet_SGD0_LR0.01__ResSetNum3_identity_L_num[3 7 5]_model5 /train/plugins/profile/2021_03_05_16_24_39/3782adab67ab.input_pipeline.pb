	??? @??? @!??? @	??~??????~????!??~????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??? @@x?=-u@1??a???b@AA?G????Is?4??%@Y??S? P??*	V-*?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator1??c??(@!?&^?X@)1??c??(@1?&^?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????O??!??x?I???)????O??1??x?I???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismg*?#????!?)?X?4??)_?L???1Q?!3???:Preprocessing2F
Iterator::Model????j???!yy????)Nd???z?1??TR??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap+N?f?(@!??4?X@)???Y.m?1???mh??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??~????In??jܐQ@Q???F?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@x?=-u@@x?=-u@!@x?=-u@      ??!       "	??a???b@??a???b@!??a???b@*      ??!       2	A?G????A?G????!A?G????:	s?4??%@s?4??%@!s?4??%@B      ??!       J	??S? P????S? P??!??S? P??R      ??!       Z	??S? P????S? P??!??S? P??b      ??!       JGPUY??~????b qn??jܐQ@y???F?=@