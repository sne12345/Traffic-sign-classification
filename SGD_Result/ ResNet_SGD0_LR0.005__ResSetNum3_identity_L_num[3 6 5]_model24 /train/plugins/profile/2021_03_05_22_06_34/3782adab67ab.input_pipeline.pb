	?h?hs0?@?h?hs0?@!?h?hs0?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?h?hs0?@?S??v@16???Вb@A?N@a???I~nh?NO@*?z????@)      p=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?$?9?@!??]'C[X@)?$?9?@1??]'C[X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch&?2????!??1?>n @)&?2????1??1?>n @:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?s????!::???=@)
?8?*??1?D?U?z??:Preprocessing2F
Iterator::Model??F???!`?Yٟ@)
??O?my?1??mh??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??KU??@!?25S_X@)??7?ܘn?1z??\g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?%]?Q@Q]??k??<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S??v@?S??v@!?S??v@      ??!       "	6???Вb@6???Вb@!6???Вb@*      ??!       2	?N@a????N@a???!?N@a???:	~nh?NO@~nh?NO@!~nh?NO@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?%]?Q@y]??k??<@