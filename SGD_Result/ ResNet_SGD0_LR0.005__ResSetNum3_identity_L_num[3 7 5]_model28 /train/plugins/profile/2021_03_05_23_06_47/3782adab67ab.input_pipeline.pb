	2;??Il?@2;??Il?@!2;??Il?@	??Q:5????Q:5??!??Q:5??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails62;??Il?@?GQgn%{@1?????b@A?Wya??I??=??$@Y??6???*	?l??y??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?9??*>)@!2ݗ}??X@)?9??*>)@12ݗ}??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch)@̘???!L?~bZ3??))@̘???1L?~bZ3??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!????7??)O??唀??1?nϽ???:Preprocessing2F
Iterator::Modell?u????!??;ڈ??)?z??~?1MɌ? |??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap(?8'@)@!??K???X@)O??D??o?1}??:7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 73.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Q:5??I???0??R@Q? ??8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?GQgn%{@?GQgn%{@!?GQgn%{@      ??!       "	?????b@?????b@!?????b@*      ??!       2	?Wya???Wya??!?Wya??:	??=??$@??=??$@!??=??$@B      ??!       J	??6?????6???!??6???R      ??!       Z	??6?????6???!??6???b      ??!       JGPUY??Q:5??b q???0??R@y? ??8@