	?n?oڻ|@?n?oڻ|@!?n?oڻ|@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?n?oڻ|@8??
j?s@1????Aa@A0.Ui?k??I6"??!@*	.????#?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorl#6@!Ұoc@?X@)l#6@1Ұoc@?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??辜??!rO??Y??)??辜??1rO??Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?qo~?D??!vY???4??)?מY???1( ?Tm??:Preprocessing2F
Iterator::Model@Û5x_??!9-?<???)??b??Հ?1??HJh??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap>?-:@!K????X@)ŏ1w-!o?1 ??s>޲?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???o??Q@Qal@n?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8??
j?s@8??
j?s@!8??
j?s@      ??!       "	????Aa@????Aa@!????Aa@*      ??!       2	0.Ui?k??0.Ui?k??!0.Ui?k??:	6"??!@6"??!@!6"??!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???o??Q@yal@n?=@