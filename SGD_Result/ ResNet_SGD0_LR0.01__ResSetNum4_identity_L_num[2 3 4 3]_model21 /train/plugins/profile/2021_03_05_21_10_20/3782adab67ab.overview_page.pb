?	?I*?@?I*?@!?I*?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?I*?@?v?Ru@1}?;l"h@AgG??????IO???((@*	?/݄??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator4g}?1?@!?M?0rX@)4g}?1?@1?M?0rX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?i??????!}MUm.'??)?i??????1}MUm.'??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????߾??!???!g @)??a????19ɲ7O???:Preprocessing2F
Iterator::Modelo*Ral!??!??)6@)V?F???x?1?B???߹?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:??@!0W?OvX@)???VC?n?1??%|z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI`?y?C.P@Q???x?A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?Ru@?v?Ru@!?v?Ru@      ??!       "	}?;l"h@}?;l"h@!}?;l"h@*      ??!       2	gG??????gG??????!gG??????:	O???((@O???((@!O???((@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`?y?C.P@y???x?A@?"9
model_25/conv2d_1343/Conv2DConv2DX??????!X??????0"j
>gradient_tape/model_25/conv2d_1349/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterذU?z???!bj?p???0"j
>gradient_tape/model_25/conv2d_1345/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltero$Y蕝?!~???a??0"j
>gradient_tape/model_25/conv2d_1352/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???/???!n%??j¿?0"h
=gradient_tape/model_25/conv2d_1348/Conv2D/Conv2DBackpropInputConv2DBackpropInputN?LR|??!\Җv????0"h
=gradient_tape/model_25/conv2d_1351/Conv2D/Conv2DBackpropInputConv2DBackpropInputst?s??!c??5???0"j
>gradient_tape/model_25/conv2d_1351/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterkN??\Ȋ?!J?셻\??0"j
>gradient_tape/model_25/conv2d_1348/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???g???!? h????0"9
model_25/conv2d_1345/Conv2DConv2D?="?Q???!?$?
????0"9
model_25/conv2d_1349/Conv2DConv2D??c?????!?_??:??0Q      Y@Y???????a?A?g?X@qT???JI@y??d??Zh?"?

both?Your program is POTENTIALLY input-bound because 62.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?50.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 