?	ᶶ???|@ᶶ???|@!ᶶ???|@	<???گ?<???گ?!<???گ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ᶶ???|@.s?,f?s@1? ?}?:_@Aɰ?72??I8?ܘ??$@Ym ]lZ)??*	?"??&??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorK?H??Z5@!~PJ??X@)K?H??Z5@1~PJ??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????"??!`???l>??)????"??1`???l>??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???m????!?є?q??)&TpxAD??1S??????:Preprocessing2F
Iterator::Model?mm?y???!???0|b??)?=?N??y?1?ښƉ,??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap%????[5@!?l?;?X@)?????q?1?	?A????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9<???گ?IaF?4#R@Q%	?>c;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.s?,f?s@.s?,f?s@!.s?,f?s@      ??!       "	? ?}?:_@? ?}?:_@!? ?}?:_@*      ??!       2	ɰ?72??ɰ?72??!ɰ?72??:	8?ܘ??$@8?ܘ??$@!8?ܘ??$@B      ??!       J	m ]lZ)??m ]lZ)??!m ]lZ)??R      ??!       Z	m ]lZ)??m ]lZ)??!m ]lZ)??b      ??!       JGPUY<???گ?b qaF?4#R@y%	?>c;@?"8
model_14/conv2d_748/Conv2DConv2D?k??%???!?k??%???0"m
Cgradient_tape/model_14/batch_normalization_748/FusedBatchNormGradV3FusedBatchNormGradV31ٿ9????!?aoً???"8
model_14/conv2d_748/BiasAddBiasAdd97]A???!IԦ??"i
=gradient_tape/model_14/conv2d_752/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterʏ?%???!QB??8C??0"8
model_14/conv2d_750/Conv2DConv2D?X|8????!?????m??0"W
1model_14/batch_normalization_748/FusedBatchNormV3FusedBatchNormV3,???`??!e m??3??"g
<gradient_tape/model_14/conv2d_757/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z:???!??f????0"8
model_14/conv2d_757/Conv2DConv2D?U???
??!m?i?8???0"8
model_14/conv2d_754/Conv2DConv2D?`{?e??!x?!KOu??0"g
<gradient_tape/model_14/conv2d_754/Conv2D/Conv2DBackpropInputConv2DBackpropInput,|O????!;?V;N4??0Q      Y@YOG9?t??a??k,?X@q?V???6@y?RFR??u?"?

both?Your program is POTENTIALLY input-bound because 70.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 