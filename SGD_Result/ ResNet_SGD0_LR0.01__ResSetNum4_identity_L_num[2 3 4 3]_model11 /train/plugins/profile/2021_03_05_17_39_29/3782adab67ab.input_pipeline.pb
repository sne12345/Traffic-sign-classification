	?	M?"}@?	M?"}@!?	M?"}@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?	M?"}@;?? *|t@1q??0c_@A??:?p??Inh?N?)@*	~?5^???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?\4d<?'@!???ʸX@)?\4d<?'@1???ʸX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??qQ-"??!??
?????)??qQ-"??1??
?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism
???ÿ?!???6?e??)?I+???1?Aj??A??:Preprocessing2F
Iterator::Model??&????!??S?$;??)?C4???y?1=!LҮ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??Y?r?'@!???l?X@)??+ٱq?1??#??D??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI,b-cDR@QO?wJs?:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?? *|t@;?? *|t@!;?? *|t@      ??!       "	q??0c_@q??0c_@!q??0c_@*      ??!       2	??:?p????:?p??!??:?p??:	nh?N?)@nh?N?)@!nh?N?)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q,b-cDR@yO?wJs?:@