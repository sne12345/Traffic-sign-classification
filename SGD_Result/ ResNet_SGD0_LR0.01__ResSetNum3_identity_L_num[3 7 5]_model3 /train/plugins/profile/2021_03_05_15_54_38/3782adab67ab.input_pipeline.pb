	fh<??@fh<??@!fh<??@	?{=??????{=?????!?{=?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6fh<??@Z????v@1?U???b@AOq??Iτ&?%?"@YY4?????*	?Q???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorB?F??q(@!?~?5?X@)B?F??q(@1?~?5?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9^??I???!??/?????)9^??I???1??/?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?$??r??!????<???)?KXc??1??þκ??:Preprocessing2F
Iterator::Model?L?????!?ASm"s??)?{??Pkz?1?)?bΪ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap}@?3is(@!|Y%??X@)-?s??m?1Ō???I??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?{=?????Id??\?Q@Q?[跓	=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z????v@Z????v@!Z????v@      ??!       "	?U???b@?U???b@!?U???b@*      ??!       2	Oq??Oq??!Oq??:	τ&?%?"@τ&?%?"@!τ&?%?"@B      ??!       J	Y4?????Y4?????!Y4?????R      ??!       Z	Y4?????Y4?????!Y4?????b      ??!       JGPUY?{=?????b qd??\?Q@y?[跓	=@