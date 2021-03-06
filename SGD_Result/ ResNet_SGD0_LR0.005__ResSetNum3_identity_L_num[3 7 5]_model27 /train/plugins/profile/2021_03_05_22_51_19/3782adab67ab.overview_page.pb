?	?ՏM???@?ՏM???@!?ՏM???@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ՏM???@????h\w@1P??;a@A;U?g$B??I?1%??.@*	C`?Т?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorm?_u?H@!?_#?ŔX@)m?_u?H@1?_#?ŔX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?h??i???!?׌g?1??)?h??i???1?׌g?1??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?7??w???!t$Q?D;??)?;ۣ7ܗ?1c3K?&??:Preprocessing2F
Iterator::Modelnߣ?z???!??Ok9???)?}:3Py?1DU?K???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?A?۽L@!?R??X@)d??1?n?1??{i+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?R?sRR@Qާ??1?:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????h\w@????h\w@!????h\w@      ??!       "	P??;a@P??;a@!P??;a@*      ??!       2	;U?g$B??;U?g$B??!;U?g$B??:	?1%??.@?1%??.@!?1%??.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?R?sRR@yާ??1?:@?"9
model_31/conv2d_1667/Conv2DConv2D??!r???!??!r???0"n
Dgradient_tape/model_31/batch_normalization_1667/FusedBatchNormGradV3FusedBatchNormGradV3.??IA??!?[?????"9
model_31/conv2d_1667/BiasAddBiasAdd?`?BzÆ?!????l??"j
>gradient_tape/model_31/conv2d_1671/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??B:????!??&??B??0"h
=gradient_tape/model_31/conv2d_1671/Conv2D/Conv2DBackpropInputConv2DBackpropInput?W?n???!?Z}????0"X
2model_31/batch_normalization_1667/FusedBatchNormV3FusedBatchNormV3x?8^??!;???????"9
model_31/conv2d_1673/Conv2DConv2DXo?W=U??!&??=<J??0"9
model_31/conv2d_1676/Conv2DConv2D??ܦaN??!l<79z??0"9
model_31/conv2d_1679/Conv2DConv2D	?F??!=???e???0"9
model_31/conv2d_1669/Conv2DConv2Dah??@??!CN??t"??0Q      Y@Y??X????a\???`?X@q?H/??pJ@y?^?R(p?"?

both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?52.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 