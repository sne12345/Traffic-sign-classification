	?ٕvK?@?ٕvK?@!?ٕvK?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ٕvK?@??b(g?t@1ū???n@AV???̯??I?}8g4 @*	?z?7??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorW???? )@!?@Qz6?X@)W???? )@1?@Qz6?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch2;?ީ??!???:>`??)2;?ީ??1???:>`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5???#b??!d	z:"??)??-Y???1?>??????:Preprocessing2F
Iterator::Model@?j?߻?!)????{??)?0DN_?w?1I???z??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???|?")@!?nj?X@)G=D?;?m?1?p????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?+?	M@Q4??o??D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??b(g?t@??b(g?t@!??b(g?t@      ??!       "	ū???n@ū???n@!ū???n@*      ??!       2	V???̯??V???̯??!V???̯??:	?}8g4 @?}8g4 @!?}8g4 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?+?	M@y4??o??D@