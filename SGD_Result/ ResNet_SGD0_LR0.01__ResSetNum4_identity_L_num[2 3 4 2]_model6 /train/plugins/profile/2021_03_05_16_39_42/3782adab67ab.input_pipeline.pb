	#??o{@#??o{@!#??o{@	?c?G]???c?G]??!?c?G]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6#??o{@ᶶ?<?s@1?E?  \@A?A??v???I??:M?$@Yw?*2: ??*	?C?lǯ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator@1?d?@!7Z??X@)@1?d?@17Z??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Rxа?!b??u?T??)???Rxа?1b??u?T??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?_?5?!??!Vi?????)??m??E??1???}???:Preprocessing2F
Iterator::Model?t ??շ?!}??V???)nR?X?;{?1m????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?-????@!???λ?X@)an?r?l?1??Vz???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 72.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?c?G]??I?Ő?p?R@Q4`???9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ᶶ?<?s@ᶶ?<?s@!ᶶ?<?s@      ??!       "	?E?  \@?E?  \@!?E?  \@*      ??!       2	?A??v????A??v???!?A??v???:	??:M?$@??:M?$@!??:M?$@B      ??!       J	w?*2: ??w?*2: ??!w?*2: ??R      ??!       Z	w?*2: ??w?*2: ??!w?*2: ??b      ??!       JGPUY?c?G]??b q?Ő?p?R@y4`???9@