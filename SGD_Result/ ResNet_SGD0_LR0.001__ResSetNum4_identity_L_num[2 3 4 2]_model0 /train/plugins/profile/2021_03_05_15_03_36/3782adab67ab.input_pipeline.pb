	???ܴ0?@???ܴ0?@!???ܴ0?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???ܴ0?@:?6U7t@1?!?
??g@AL?Qԙ??I<?$ @*	?(\?(?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorϢw*?N@!??&	??X@)Ϣw*?N@1??&	??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?_!se??!????pI??)?_!se??1????pI??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????N??!?e?? ???)?)??z???1?_??"??:Preprocessing2F
Iterator::Model?]L3????!>?δeF??)Pqx??y?1?
??RD??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap)@̘R@!'?,i??X@)?????m?1621 c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI-t3`[?O@QӋ̟?*B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:?6U7t@:?6U7t@!:?6U7t@      ??!       "	?!?
??g@?!?
??g@!?!?
??g@*      ??!       2	L?Qԙ??L?Qԙ??!L?Qԙ??:	<?$ @<?$ @!<?$ @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q-t3`[?O@yӋ̟?*B@