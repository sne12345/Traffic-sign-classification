	vŌ???}@vŌ???}@!vŌ???}@	?tu??v???tu??v??!?tu??v??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6vŌ???}@??6Zs@1~W??d@A`=?[???IDOʤF%@YZ??c!??*	1?Zj?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator1??*?(@!?ᰣh?V@)1??*?(@1?ᰣh?V@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??s|????!kJ??!? @)??s|????1kJ??!? @:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism.c}S??!Ւ???? @)-??;????1???????:Preprocessing2F
Iterator::Model????5q??!M??W?!@)Lo.2~?1?w??m???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???M?(@!v?Տ?V@)?&S?r?1?d??9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 63.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?tu??v??I%?/#ЋP@QX?????@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??6Zs@??6Zs@!??6Zs@      ??!       "	~W??d@~W??d@!~W??d@*      ??!       2	`=?[???`=?[???!`=?[???:	DOʤF%@DOʤF%@!DOʤF%@B      ??!       J	Z??c!??Z??c!??!Z??c!??R      ??!       Z	Z??c!??Z??c!??!Z??c!??b      ??!       JGPUY?tu??v??b q%?/#ЋP@yX?????@@