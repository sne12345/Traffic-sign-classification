?	2;??Il?@2;??Il?@!2;??Il?@	??Q:5????Q:5??!??Q:5??"w
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
	?GQgn%{@?GQgn%{@!?GQgn%{@      ??!       "	?????b@?????b@!?????b@*      ??!       2	?Wya???Wya??!?Wya??:	??=??$@??=??$@!??=??$@B      ??!       J	??6?????6???!??6???R      ??!       Z	??6?????6???!??6???b      ??!       JGPUY??Q:5??b q???0??R@y? ??8@?"9
model_32/conv2d_1725/Conv2DConv2D괤???!괤???0"n
Dgradient_tape/model_32/batch_normalization_1725/FusedBatchNormGradV3FusedBatchNormGradV3?!??B??!x??????"9
model_32/conv2d_1725/BiasAddBiasAddJ1?8???!?C?Z^??"h
=gradient_tape/model_32/conv2d_1731/Conv2D/Conv2DBackpropInputConv2DBackpropInputu?s?{??!?Ɗ????0"9
model_32/conv2d_1727/Conv2DConv2D?Rąk??!?P_H???0"X
2model_32/batch_normalization_1725/FusedBatchNormV3FusedBatchNormV3P?j\j??!U4f??h??"9
model_32/conv2d_1734/Conv2DConv2D?dtt'h??!???ژ??0"9
model_32/conv2d_1737/Conv2DConv2Do-7?f??!Q?-O<a??0"9
model_32/conv2d_1731/Conv2DConv2D<
??ub??!??u?c???0"h
=gradient_tape/model_32/conv2d_1737/Conv2D/Conv2DBackpropInputConv2DBackpropInput1?^?W??!?a]???0Q      Y@Y??X????a\???`?X@q?-0??A@y???p?"?

both?Your program is POTENTIALLY input-bound because 73.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?35.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 