	?V*(??@?V*(??@!?V*(??@	????ꪚ?????ꪚ?!????ꪚ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?V*(??@?A?????@1&?ls?2o@A??q?߅??In4??@R@YʉvR~??*	????댹@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Z(???@!g?????X@)?Z(???@1g?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchDܜJ???!???La???)DܜJ???1???La???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismaQ??l??!`??&K??)t??q5???1LW h?v??:Preprocessing2F
Iterator::Model?3??k???!??ݹ??)?$>w??w?1?$?pZ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap̘?5?@!??ǈ?X@)??K?l?12T????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 71.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????ꪚ?I2??( ,R@Q?q???H;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?A?????@?A?????@!?A?????@      ??!       "	&?ls?2o@&?ls?2o@!&?ls?2o@*      ??!       2	??q?߅????q?߅??!??q?߅??:	n4??@R@n4??@R@!n4??@R@B      ??!       J	ʉvR~??ʉvR~??!ʉvR~??R      ??!       Z	ʉvR~??ʉvR~??!ʉvR~??b      ??!       JGPUY????ꪚ?b q2??( ,R@y?q???H;@